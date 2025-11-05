# INGP / Laplacian Pyramids Comparison

# Installation

```bash
git clone git@github.com:Tetchki/ingp-pyramid-comparaison.git
cd ingp-pyramid-comparaison
pip install -r requirements.txt
```

# Usage

In the src directory, run:

```bash
python3 scene_opti.py --config ../scenes/painting/painting.gin --method both
```

Or to run a texture optimization:

```bash
python3 texture_opti.py --ref ../scenes/textures/The_Great_Wave_off_Kanagawa_4k.jpg --method both --lr 1e-3 --resolution 16 --iterations 512 --output_path ../results/texture_opti
```

# Modification

You can modify the parameters of the INGP and Laplacian Pyramid in their respective XML files
(e.g. `scenes/painting/painting_neural.xml` for INGP and `scenes/painting/painting_pyramid.xml` for Laplacian Pyramid).

The rest of the parameters (lr, spp, etc.) can be changed in the gin config file (e.g. `scenes/painting/painting.gin` ).
