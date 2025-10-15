import argparse
import dataclasses
import functools
import time
import os
from typing import Optional, Dict, Any

import mitsuba as mi
import drjit as dr
from drjit.auto.ad import Float16, Float32
from drjit.opt import GradScaler
import gin
from tqdm import trange
import matplotlib.pyplot as plt

from config import SceneConfig
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run INGP method on a given scene.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gin configuration file.")
    args = parser.parse_args()
    config = os.path.abspath(args.config)
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Configuration file not found: {config}")
    return config

def run_ingp(scene_config: SceneConfig):
    absolute_path = os.path.abspath(scene_config.scene_path)
    print(f"Running INGP method on scene: {absolute_path}")

    intermediate_output_path = os.path.join(scene_config.output_path, "frames")
    losses_path = os.path.join(scene_config.output_path, "losses")

    mi.set_variant("cuda_ad_rgb")

    ref_image = mi.render(mi.load_file(scene_config.ref_image_path), spp=scene_config.spp)

    save_image(ref_image, scene_config.output_path, "reference", format='exr')

    import neural_texture
    mi.register_texture("neuraltexture", lambda props: neural_texture.NeuralTexture(props))

    scene = mi.load_file(scene_config.scene_path, optimize=False)

    params = mi.traverse(scene)

    opti_params, encoding_params, network_weights = extract_weights_and_encoding(params)

    opt = mi.ad.Adam(lr=scene_config.lr, params=opti_params, beta_1=scene_config.beta_1, beta_2=scene_config.effective_beta_2)

    scaler = GradScaler()

    start_time = time.time()
    losses = []

    print("Starting optimization...")

    pbar = trange(scene_config.iterations, desc="Optimizing", ncols=100)

    for i in pbar:

        for k, nw in zip(
                [key for key in opt.keys() if key.endswith("network_weights")],
                network_weights
        ):
            if opt.get(k) is not None:
                nw[:] = Float16(opt[k])

        for k, ep in zip(
                [key for key in opt.keys() if key.endswith("encoding_params")],
                encoding_params
        ):
            if opt.get(k) is not None:
                ep[:] = Float16(opt[k])

        #dr.profile_mark("render")
        image = mi.render(scene, params, spp=scene_config.spp_primal, spp_grad=scene_config.spp_grad)

        clean = lambda x: dr.select(dr.isfinite(x), x, 0.0)
        image_s = clean(image)
        reference_s = clean(ref_image)

        #dr.profile_mark("loss")
        loss = dr.mean(dr.square(image_s - reference_s))

        #dr.profile_mark("backward")
        dr.backward(scaler.scale(loss))
        #dr.profile_mark("step")
        scaler.step(opt)

        if float(loss.array[0]) > 100:
            save_image(image, scene_config.output_path, f"iter_{i:04d}", format='exr')
            raise RuntimeError(f"Loss diverged: {float(loss.array[0]):.6f}")

        pbar.set_description(f"Iter {i:02d} | lr={scene_config.lr:.4g} | loss={float(loss.array[0]):.6f}")
        losses.append(float(loss.array[0]))

        if scene_config.save_interval > 0 and (i % scene_config.save_interval == 0 or i == scene_config.iterations - 1):
            save_image(image, intermediate_output_path, f"iter_{i:04d}", format='exr')

    end_time = time.time()
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds.")

    # Render final image
    if scene_config.rerender_spp > 0:
        print(f"Rerendering final image with {scene_config.rerender_spp} spp...")
        final_image = mi.render(scene, params, spp=scene_config.rerender_spp)
        save_image(final_image, scene_config.output_path, f"final_image_{scene_config.rerender_spp}spp", format='exr')

    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Loss Over Iterations')
    plt.grid(True)
    if not os.path.exists(losses_path):
        os.makedirs(losses_path)
    plt.savefig(os.path.join(losses_path, 'loss_plot.png'))
    plt.close()

    np.save(os.path.join(losses_path, 'losses.npy'), np.array(losses))

def extract_weights_and_encoding(params):
    network_weights_keys = [k for k in params.keys() if k.endswith('.network_weights')]
    encoding_params_keys = [k for k in params.keys() if k.endswith('.encoding_params')]

    optimizer_parameters = {}

    network_weights = []
    encoding_params = []

    # Collect all network_weights
    for k in network_weights_keys:
        val = params[k]
        network_weights.append(val)
        optimizer_parameters[k] = Float32(val)

    # Collect all encoding_params
    for k in encoding_params_keys:
        val = params[k]
        encoding_params.append(val)
        optimizer_parameters[k] = Float32(val)

    return optimizer_parameters, encoding_params, network_weights

def load_scene_config(config_path: str) -> SceneConfig:
    gin.clear_config()
    gin.parse_config_files_and_bindings([config_path], None, skip_unknown=False)
    cfg = SceneConfig()
    cfg.validate()
    return cfg

def save_image(image, path, name, format='exr'):

    if not os.path.exists(path):
        os.makedirs(path)

    mi.util.write_bitmap(
        os.path.join(path, f"{name}.{format}"),
        image
    )

if __name__ == "__main__":

    config_path = parse_arguments()
    scene_cfg = load_scene_config(config_path)

    run_ingp(scene_cfg)





