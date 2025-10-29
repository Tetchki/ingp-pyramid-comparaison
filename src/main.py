from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import mitsuba as mi
import drjit as dr
from drjit.opt import GradScaler, Optimizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from util.config import SceneConfig
from pyramid import flatmip_aware_pyramid
from src.ingp import instantNGP
from util import parameters as parameters_lib
from util.parameters import MitsubaVariables

class Method(str, Enum):
    BOTH = "both"
    INGP = "ingp"
    PYRAMID = "pyramid"

def parse_arguments() -> Tuple[Path, Method]:
    """Parse CLI arguments and validate the config path."""
    parser = argparse.ArgumentParser(description="Run INGP/Pyramid optimization on a scene.")
    parser.add_argument("--config", type=str, required=True, help="Path to a gin config file.")
    parser.add_argument(
        "--method",
        type=str,
        default=Method.BOTH.value,
        choices=[m.value for m in Method],
        help="Method to use: 'ingp', 'pyramid', or 'both' for a comparative run.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return config_path, Method(args.method)

def setup_logging() -> None:
    """Configure logging once for the whole script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def load_scene_config(config_path: Path) -> SceneConfig:
    """Load and validate the gin-based SceneConfig."""
    gin.clear_config()
    gin.parse_config_files_and_bindings([str(config_path)], None, skip_unknown=False)
    cfg = SceneConfig()
    cfg.validate()
    return cfg

def configure_scene(method: Method, scene_cfg: SceneConfig) -> mi.Scene:
    """
    Prepare Mitsuba scene according to the chosen method.
    Also manages module imports and texture registrations.
    Args:
        method: Method enum indicating INGP or PYRAMID.
        scene_cfg: SceneConfig instance with scene file paths.
    Returns:
        scene: The mitsuba.Scene instance to optimize.
    """
    if method == Method.INGP:
        try:
            from ingp import neural_texture
        except Exception as e:
            raise RuntimeError("Failed to import 'neural_texture' required for INGP.") from e
        mi.register_texture("neuraltexture", lambda props: neural_texture.NeuralTexture(props))
        scene = mi.load_file(scene_cfg.ingp_scene, optimize=False)
    elif method == Method.PYRAMID:
        try:
            from pyramid import mipmap_flat
        except Exception as e:
            raise RuntimeError("Failed to import 'mipmap_flat' required for pyramid.") from e
        mipmap_flat.register()
        scene = mi.load_file(scene_cfg.pyramid_scene, optimize=False)
    else:
        raise ValueError(f"Unsupported method for configure_scene: {method}")
    return scene

def build_optimizer(scene_cfg: SceneConfig) -> mi.ad.Optimizer:
    """Create and return a Mitsuba Adam optimizer with the configured hyperparams.
    Args:
        scene_cfg: SceneConfig instance with optimization hyperparameters.
    Returns:
        An instance of mitsuba.ad.Optimizer (Adam).
    """
    return mi.ad.Adam(
        lr=scene_cfg.lr,
        beta_1=scene_cfg.beta_1,
        beta_2=scene_cfg.effective_beta_2,
    )

def build_variables_for_ingp(params: mi.python.util.SceneParameters,
                             scene_cfg: SceneConfig) -> tuple[MitsubaVariables, Optimizer]:
    """Collect INGP variables
    Args:
        params: The scene parameters
        scene_cfg: SceneConfig instance with optimization hyperparameters.
    Returns:
        A tuple of (MitsubaVariables, Optimizer).
    """
    object_names = extract_ingp_objects_from_scene_parameters(params)

    instant_ngp_vars: List[instantNGP.InstantNGPVariable] = []
    opt = build_optimizer(scene_cfg)

    for obj_name in object_names:
        instant_ngp_vars.append(
            instantNGP.InstantNGPVariable(
                key=obj_name,
                optimizer=f"{obj_name}",
                initial_value=0,
            )
        )

    variables = parameters_lib.MitsubaVariables(instant_ngp_vars, params)
    variables.initialize(opt)
    return variables, opt

def build_variables_for_pyramid(params: mi.python.util.SceneParameters,
                                scene_cfg: SceneConfig) -> tuple[MitsubaVariables, Optimizer]:
    """Collect pyramid variables
    Args:
        params: The scene parameters
        scene_cfg: SceneConfig instance with optimization hyperparameters.
    Returns:
        A tuple of (MitsubaVariables, Optimizer).
    """
    pyramid_params = extract_pyramid_parameters_from_scene_parameters(params)

    pyramid_vars: List[flatmip_aware_pyramid.FlatMipAwareImagePyramidVariable] = []
    opt = build_optimizer(scene_cfg)

    for obj_name, p_params in pyramid_params.items():
        pyramid_vars.append(
            flatmip_aware_pyramid.FlatMipAwareImagePyramidVariable(
                key=f"{obj_name}.data",
                optimizer=f"{obj_name}.data.flat_buffer",
                initial_value=0,
                n_levels=p_params["n_levels"],
                factor=p_params["mip_factor"],
                shape=None,
                mipmapped=True,
                normal_clamping=False,
                ensure_frequency_decomposition=False,
                learning_rate=scene_cfg.lr,
                is_scene_parameter=True,
            )
        )

    variables = parameters_lib.MitsubaVariables(pyramid_vars, params)
    variables.initialize(opt)
    return variables, opt

def save_image(image, out_dir: Path, name: str, fmt: str = "exr") -> None:
    """Write an image to disk, creating the directory if needed.
    Args:
        image: The Mitsuba image to save.
        out_dir: The output directory path.
        name: The base name for the saved image file.
        fmt: The image format/extension (default: 'exr').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(out_dir / f"{name}.{fmt}"), image)

def plot_and_save_losses(losses: List[float], out_dir: Path) -> None:
    """Save a log-scale loss plot and the raw losses array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Loss Over Iterations")
    plt.grid(True)
    plt.savefig(out_dir / "loss_plot.png")
    plt.close()
    np.save(out_dir / "losses.npy", np.array(losses))

def extract_ingp_objects_from_scene_parameters(params: mi.python.util.SceneParameters) -> List[str]:
    """
    Extract object names that have INGP encoding parameters.
    Args:
        params: The scene parameters.
    Returns:
        All object names with INGP encodings.
    """
    object_keys = [k for k in params.keys() if k.endswith(".encoding_params")]
    return [k.replace(".encoding_params", "") for k in object_keys]

def extract_pyramid_parameters_from_scene_parameters(
    scene_parameters: mi.python.util.SceneParameters,
) -> Dict[str, Dict[str, object]]:
    """
    Collect pyramid-related parameters per object name.
    Args:
        scene_parameters: The scene parameters.
    Returns:
        A dictionary mapping object names to their pyramid parameters.
    """
    object_keys = [k for k in scene_parameters.keys() if k.endswith("data.base_mip_shape")]
    object_names = [k.replace(".data.base_mip_shape", "") for k in object_keys]

    pyramid_params: Dict[str, Dict[str, object]] = {}

    for obj_name in object_names:
        base_mip_shape = scene_parameters[f"{obj_name}.data.base_mip_shape"]
        mip_factor = scene_parameters[f"{obj_name}.data.mip_factor"]
        flat_buffer = scene_parameters[f"{obj_name}.data.flat_buffer"]
        flat_buffer_offsets = scene_parameters[f"{obj_name}.data.flat_buffer_offsets"]

        n_levels = len(flat_buffer_offsets) - 1
        if base_mip_shape[2] == 1:
            storage_type = mi.Float
        else:
            assert base_mip_shape[2] == 3, "Unsupported channel count for pyramid storage."
            storage_type = mi.Color3f

        pyramid_params[obj_name] = {
            "base_mip_shape": base_mip_shape,
            "mip_factor": mip_factor,
            "flat_buffer": flat_buffer,
            "flat_buffer_offsets": flat_buffer_offsets,
            "n_levels": n_levels,
            "storage_type": storage_type,
        }
    return pyramid_params

@dataclass
class OptimizationPaths:
    output: Path
    frames: Path
    losses: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "OptimizationPaths":
        return cls(output=output_dir, frames=output_dir / "frames", losses=output_dir / "losses")


def optimize_once(scene_cfg: SceneConfig, method: Method) -> List[float]:
    """
    Run a single optimization for the given method.
    Args:
        scene_cfg: SceneConfig instance with scene and optimization parameters.
        method: Method enum indicating INGP or PYRAMID.
    """
    method_output_dir = Path(scene_cfg.output_path) / method.value
    out_paths = OptimizationPaths.from_output_dir(method_output_dir)

    mi.set_variant("cuda_ad_rgb")

    ref_scene_path = Path(scene_cfg.ref_scene).resolve()
    logging.info("Running %s on scene: %s", method.value, ref_scene_path)

    ref_image = mi.render(mi.load_file(scene_cfg.ref_scene), spp=scene_cfg.ref_spp)
    save_image(ref_image, out_paths.output, "reference", fmt="exr")

    scene = configure_scene(method, scene_cfg)
    params = mi.traverse(scene)

    scaler = GradScaler()

    if method == Method.INGP:
        variables, opt = build_variables_for_ingp(params, scene_cfg)
    elif method == Method.PYRAMID:
        variables, opt = build_variables_for_pyramid(params, scene_cfg)
    else:
        raise ValueError(f"Unsupported method for optimize_once: {method}")

    logging.info("Optimizing the following parameters:")
    for key in opt.keys():
        logging.info("  - %s", key)

    losses: List[float] = []
    seed = 0
    start = time.time()
    logging.info("Starting optimization for %d iterations...", scene_cfg.iterations)
    pbar = trange(scene_cfg.iterations, desc="Optimizing", ncols=100)

    for i in pbar:
        image = mi.render(
            scene,
            params,
            spp=scene_cfg.spp_primal,
            spp_grad=scene_cfg.spp_grad,
            seed=seed,
        )
        seed += 1

        loss = dr.mean(dr.square(image - ref_image))
        dr.backward(scaler.scale(loss))

        variables.process_gradients(opt)
        scaler.step(opt)
        variables.update(opt, i)

        loss_val = float(loss.array[0])
        pbar.set_description(f"Iter {i:04d} | lr={scene_cfg.lr:.4g} | loss={loss_val:.6f}")
        losses.append(loss_val)

        if scene_cfg.save_interval > 0 and (i % scene_cfg.save_interval == 0 or i == scene_cfg.iterations - 1):
            save_image(image, out_paths.frames, f"iter_{i:04d}", fmt="exr")

    elapsed = time.time() - start
    logging.info("Optimization completed in %.2f seconds.", elapsed)

    if scene_cfg.rerender_spp > 0:
        logging.info("Rerendering final image with %d spp...", scene_cfg.rerender_spp)
        final_image = mi.render(scene, params, spp=scene_cfg.rerender_spp)
        save_image(final_image, out_paths.output, f"final_image_{scene_cfg.rerender_spp}spp", fmt="exr")

    plot_and_save_losses(losses, out_paths.losses)

    return losses

def plot_comparison(ingp_losses: List[float], pyramid_losses: List[float], scene_cfg: SceneConfig) -> None:
    """Plot and save a comparison of losses between INGP and Pyramid methods."""
    comparison_path = Path(scene_cfg.output_path + "/comparison_loss_plot.png")

    plt.figure(figsize=(10, 6))
    plt.plot(ingp_losses, label="INGP Loss", color="blue")
    plt.plot(pyramid_losses, label="Pyramid Loss", color="orange")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Comparison between INGP and Pyramid Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(comparison_path)
    plt.close()

def main():
    setup_logging()
    config_path, method = parse_arguments()
    scene_cfg = load_scene_config(config_path)

    if method == Method.BOTH:
        ingp_losses = optimize_once(scene_cfg, Method.INGP)
        pyramid_losses = optimize_once(scene_cfg, Method.PYRAMID)

        plot_comparison(ingp_losses, pyramid_losses, scene_cfg)
    else:
        optimize_once(scene_cfg, method)

if __name__ == "__main__":
    main()
