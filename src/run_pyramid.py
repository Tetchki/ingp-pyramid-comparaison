import argparse
import os
import sys

import mitsuba as mi

from practical_reconstruction import optimization_cli
from core import integrators
from core import bsdfs
from core import textures

_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Pyramid method on a given scene.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gin configuration file.")
    args = parser.parse_args()
    config = os.path.abspath(args.config)
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Configuration file not found: {config}")
    return config

if __name__ == "__main__":

    # Disable LaTeX rendering in matplotlib
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False
    mi.set_variant('cuda_ad_rgb')

    config_path = parse_arguments()

    integrators.register()
    bsdfs.register()
    textures.register()

    # Clears tmp folder from previous runs
    if os.path.exists(os.path.join(_ROOT_PATH, "tmp")):
        import shutil
        shutil.rmtree(os.path.join(_ROOT_PATH, "tmp"))
        print("Cleared tmp folder from previous runs")

    gin_config_name = config_path
    # run_config expects the config name without the .gin suffix
    if gin_config_name.endswith('.gin'):
        gin_config_name = gin_config_name[:-4]
    override_bindings = []

    optimization_cli.run_config(gin_config_name, override_bindings)




