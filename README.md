# INGP / Laplacian Pyramids Comparison

# Installation

```bash
git clone --recurse-submodules git@github.com:Tetchki/ingp-pyramid-comparaison.git
cd ingp-pyramid-comparaison
pip install -r requirements.txt
pip install -e .
```

# Usage

In the src directory, run:

```bash
python run_ingp.py --config ../scenes/painting/INGP/painting.gin
python run_pyramid.py --config ../scenes/painting/Pyramid/painting_base.gin
```

# Modification

To modify the hashgrid's structure, change the parameters in the xml file (e.g. `scenes/painting/INGP/painting_neural.xml`).

The rest of the parameters (lr, spp, etc.) can be changed for both methods in their respective gin files (e.g. `scenes/painting/INGP/painting.gin` and `scenes/painting/Pyramid/painting_base.gin`).
