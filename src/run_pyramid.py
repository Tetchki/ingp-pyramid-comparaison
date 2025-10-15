import os
import sys

import mitsuba as mi

from practical_reconstruction import optimization_cli
from core import integrators
from core import bsdfs
from core import textures

_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":

    # Disable LaTeX rendering in matplotlib
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False
    mi.set_variant('cuda_ad_rgb')

    integrators.register()
    bsdfs.register()
    textures.register()

    # Clears tmp folder from previous runs
    if os.path.exists(os.path.join(_ROOT_PATH, "tmp")):
        import shutil
        shutil.rmtree(os.path.join(_ROOT_PATH, "tmp"))
        print("Cleared tmp folder from previous runs")


    gin_config_name = "../scenes/painting/Pyramid/painting_base"
    override_bindings = []

    optimization_cli.run_config(gin_config_name, override_bindings)




