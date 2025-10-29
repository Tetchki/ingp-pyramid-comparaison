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
python3 main.py --config ../scenes/painting/painting.gin --method both
```

# Modification

To modify the hashgrid's structure, change the parameters in the xml file (e.g. `scenes/painting/painting_neural.xml`).

The rest of the parameters (lr, spp, etc.) can be changed in the gin config file (e.g. `scenes/painting/painting.gin` ).
