from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict

import mitsuba as mi
import drjit as dr
from drjit.opt import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import imageio.v3 as iio

import pyramid.pyramid
from pyramid.mipmap_flat import MipmapFlatBitmap
from pyramid import flatmip_aware_pyramid
from hashgrid import hashgrid, neural_texture
from util import parameters as parameters_lib
from util.parameters import MitsubaVariables

from drjit.auto.ad import Texture2f, TensorXf, Float32, Array2f


class Method(str, Enum):
    BOTH = "both"
    HASHGRID = "hashgrid"
    PYRAMID = "pyramid"


def parse_arguments() -> Tuple[Path, Method, float, int, int, Path]:
    """
    Parse CLI args for a reference image optimization run.

    Returns:
        (ref_path, method, lr, iterations, resolution, output_path)
    """
    parser = argparse.ArgumentParser(description="Run HashGrid/Pyramid optimization on a texture.")
    parser.add_argument("--ref", type=str, help="Path to the reference image.", required=True)
    parser.add_argument(
        "--method",
        type=str,
        default=Method.BOTH.value,
        choices=[m.value for m in Method],
        help="Method to use: 'hashgrid', 'pyramid', or 'both' for a comparative run.",
    )
    parser.add_argument("--lr", type=float, help="Learning rate.", default=1e-3)
    parser.add_argument("--resolution", type=int, help="Minimum texture resolution.", default=16)
    parser.add_argument("--iterations", type=int, help="Number of optimization iterations.", default=128)
    parser.add_argument("--output_path", type=str, help="Output directory.", default="../results/")

    args = parser.parse_args()
    return Path(args.ref), Method(args.method), args.lr, args.iterations, args.resolution, Path(args.output_path)


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def build_variables_for_hashgrid(
    resolution: int, texture_shape, lr: float
) -> tuple[MitsubaVariables, mi.ad.Optimizer, mi.Texture]:
    """
    Create a neural hashgrid texture and wrap its parameters for optimization.

    Args:
        resolution: Minimum base resolution for the hashgrid encoding.
        texture_shape: Shape (H, W, C) of the reference texture.
        lr: Optimizer learning rate.

    Returns:
        (variables, optimizer, texture)
    """
    levels = pyramid.pyramid.compute_n_levels(shape=texture_shape, factor=2, min_resolution=resolution)
    hashmap_size = texture_shape[0] * texture_shape[1] * texture_shape[2]

    opt = mi.ad.Adam(lr=lr)

    # Define variables expected by our parameter management layer
    instant_ngp_vars: List[hashgrid.hashgridVariable] = []
    obj_name = ""
    instant_ngp_vars.append(
        hashgrid.hashgridVariable(
            key=obj_name,
            optimizer=obj_name,
            initial_value=0,
        )
    )
    logging.info("  - Hashgrid with %d levels", levels)

    # Register and instantiate the Mitsuba texture
    mi.register_texture("neuraltexture", lambda props: neural_texture.NeuralTexture(props))
    texture = mi.load_dict(
        {
            "type": "neuraltexture",
            "encoding_type": "hashgrid",
            "n_levels": levels,
            "n_features_per_level": 3,
            "hashmap_size": hashmap_size,
            "base_resolution": resolution,
            "per_level_scale": 2,
        }
    )

    params = mi.traverse(texture)
    variables = parameters_lib.MitsubaVariables(instant_ngp_vars, params)
    variables.initialize(opt)
    return variables, opt, texture


def build_variables_for_pyramid(
    resolution: int, texture_shape, lr: float, filepath: Path
) -> tuple[MitsubaVariables, mi.ad.Optimizer, mi.Texture]:
    """
    Create a flat mip-aware image pyramid texture initialized from an image
    and wrap its parameters for optimization.

    Args:
        resolution: Minimum base resolution of the pyramid.
        texture_shape: Shape (H, W, C) of the reference texture.
        lr: Optimizer learning rate.
        filepath: Path to the image file used to initialize the pyramid.

    Returns:
        (variables, optimizer, texture)
    """
    levels = pyramid.pyramid.compute_n_levels(shape=texture_shape, factor=2, min_resolution=resolution)

    pyramid_vars: List[flatmip_aware_pyramid.FlatMipAwareImagePyramidVariable] = []
    opt = mi.ad.Adam(lr=lr)

    obj_name = ""
    pyramid_vars.append(
        flatmip_aware_pyramid.FlatMipAwareImagePyramidVariable(
            key=f"{obj_name}.data",
            optimizer=f"{obj_name}.data.flat_buffer",
            initial_value=0,
            n_levels=levels,
            factor=-1,
            shape=None,
            mipmapped=True,
            normal_clamping=False,
            ensure_frequency_decomposition=False,
            learning_rate=lr,
            is_scene_parameter=True,
        )
    )

    mi.register_texture("mipmap_flat", lambda props: MipmapFlatBitmap(props))
    texture = mi.load_dict(
        {
            "type": "mipmap_flat",
            # Simply used to extract the shape and number of channels
            "nested_bitmap": {
                "type": "bitmap",
                "format": "variant",
                "filename": str(filepath),
            },
            "filter_type": "gaussian",
            "downsampling_factor": 2,
            "min_resolution": resolution,
            "volume_albedo_remap": False,
            "border_mode": "repeat",
            "mip_bias": 0.0,
        }
    )

    params = mi.traverse(texture)
    logging.info("  - Pyramid with %d levels", levels)

    variables = parameters_lib.MitsubaVariables(pyramid_vars, params)
    variables.initialize(opt)
    return variables, opt, texture


def save_image(image, out_dir: Path, name: str, fmt: str = "exr") -> None:
    """
    Write a Mitsuba bitmap or tensor image to disk.

    Args:
        image: Image to save (compatible with mi.util.write_bitmap).
        out_dir: Output directory.
        name: Base filename without extension.
        fmt: File extension/format (default: 'exr').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(out_dir / f"{name}.{fmt}"), image)


def plot_and_save_losses(losses: List[float], out_dir: Path) -> None:
    """
    Save a loss plot (linear y-scale) and the raw losses array.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Loss Over Iterations")
    plt.grid(True)
    plt.savefig(out_dir / "loss_plot.png")
    plt.close()
    np.save(out_dir / "losses.npy", np.array(losses))


@dataclass
class OptimizationPaths:
    output: Path
    frames: Path
    losses: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "OptimizationPaths":
        return cls(output=output_dir, frames=output_dir / "frames", losses=output_dir / "losses")


def optimize_once(
    ref_path: Path, output_path: Path, method: Method, lr: float, iterations: int, resolution: int
) -> List[float]:
    """
    Run one optimization pass for a chosen method (hashgrid or pyramid) to
    approximate a reference image by optimizing the texture parameters.

    Args:
        ref_path: Path to the reference image.
        output_path: Root directory for outputs.
        method: Optimization backend (HASHGRID or PYRAMID).
        lr: Learning rate.
        iterations: Number of optimization steps.
        resolution: Minimum resolution for the texture representation.

    Returns:
        List of per-iteration loss values.
    """
    method_output_dir = Path(output_path) / method.value
    out_paths = OptimizationPaths.from_output_dir(method_output_dir)

    mi.set_variant("cuda_ad_rgb")

    # Load reference image (range [0,1])
    ref = TensorXf(iio.imread(ref_path) / 255.0)
    shape = ref.shape
    assert shape[0] == shape[1], "Reference image must be square."
    ref_tex = Texture2f(ref)

    logging.info("Running %s on: %s", method.value, ref_path)

    rng = dr.rng(seed=0)

    logging.info("Optimizing the following parameters:")
    if method == Method.HASHGRID:
        variables, opt, texture = build_variables_for_hashgrid(resolution, shape, lr)
    elif method == Method.PYRAMID:
        variables, opt, texture = build_variables_for_pyramid(resolution, shape, lr, ref_path)
    else:
        raise ValueError(f"Unsupported method for optimize_once: {method}")

    losses: List[float] = []
    seed = 0
    start = time.time()
    logging.info("Starting optimization for %d iterations...", iterations)
    pbar = trange(iterations, desc="Optimizing", ncols=100)

    for i in pbar:
        res = shape[0]
        t = dr.arange(Float32, res)
        p = (Array2f(dr.meshgrid(t, t)) + rng.random(Array2f, (2, res * res))) / res

        si = dr.zeros(mi.SurfaceInteraction3f, 1)
        si.uv.x = p[:, 0]
        si.uv.y = 1.0 - p[:, 1]

        seed += 1

        loss = dr.mean(dr.square(ref_tex.eval(p) - texture.eval(si)))
        dr.backward(loss)

        variables.process_gradients(opt)
        opt.step()
        variables.update(opt, i)

        loss_val = float(loss.array[0])
        pbar.set_description(f"Iter {i:04d} | lr={lr:.4g} | loss={loss_val:.6f}")
        losses.append(loss_val)

    elapsed = time.time() - start
    logging.info("Optimization completed in %.2f seconds.", elapsed)

    plot_and_save_losses(losses, out_paths.losses)

    # Export final texture and reference as EXR
    res = shape[0]
    t = dr.linspace(Float32, 0, 1, res)
    p = Array2f(dr.meshgrid(t, t))

    si = dr.zeros(mi.SurfaceInteraction3f, 1)
    si.uv.x = p[:, 0]
    si.uv.y = 1.0 - p[:, 1]

    img = texture.eval(si)
    img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))
    save_image(img, out_paths.output, "optimized_texture", fmt="exr")
    save_image(ref, out_paths.output, "reference", fmt="exr")

    return losses


def plot_comparison(hashgrid_losses: List[float], pyramid_losses: List[float], output_path: Path) -> None:
    """
    Plot and save a comparison of losses between the Hashgrid and Pyramid methods.
    """
    comparison_path = output_path / "loss_comparison.png"
    plt.figure(figsize=(10, 6))
    plt.plot(hashgrid_losses, label="Hashgrid Loss")
    plt.plot(pyramid_losses, label="Pyramid Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Comparison between Hashgrid and Pyramid Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(comparison_path)
    plt.close()


def main():
    setup_logging()
    ref_path, method, lr, iterations, resolution, output_path = parse_arguments()

    if method == Method.BOTH:
        ingp_losses = optimize_once(ref_path, output_path, Method.HASHGRID, lr, iterations, resolution)
        pyramid_losses = optimize_once(ref_path, output_path, Method.PYRAMID, lr, iterations, resolution)
        plot_comparison(ingp_losses, pyramid_losses, output_path)
    else:
        optimize_once(ref_path, output_path, method, lr, iterations, resolution)


if __name__ == "__main__":
    main()