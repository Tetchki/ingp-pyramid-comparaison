import mitsuba as mi
import drjit as dr
import drjit.nn as nn

class NeuralTexture(mi.Texture):
    """
        Neural texture (:monosp:`neuraltexture`)

        A differentiable texture backed by a small network with configurable
        input encodings. It maps UVs → RGB using one of:
        - 'hashgrid' / 'permuto' (multi-resolution encodings)
        - 'tri' / 'sin' (triangular / sinusoidal encodings)

        Attributes:
            encoding_type (str): The type of encoding to use. Options are:
                - 'hashgrid' / 'permuto' for multi-resolution encodings.
                - 'tri' / 'sin' for triangular or sinusoidal encodings.
            n_levels (int): Number of levels for hashgrid/permuto encodings.
            n_features_per_level (int): Features per level for hashgrid/permuto encodings.
            base_resolution (int): Base resolution for hashgrid/permuto encodings.
            per_level_scale (float): Scale factor per level for hashgrid/permuto encodings.
            hashmap_size (int): Size of the hashmap for hashgrid/permuto encodings.
            octaves (int): Number of octaves for tri/sin encodings.
            shift (float): Shift parameter for tri/sin encodings.
            hidden_size (int): Width of the hidden layers in the MLP. If 0, no MLP is used.
            num_layers (int): Depth of the MLP. If 0, no MLP is used.
            dtype (str): Precision of the network. Options are 'float32', 'float16', or 'float8'.

        Usage:
            <texture type="neuraltexture" name="base_color">
			    <string name="encoding_type" value="hashgrid"/>
				  <integer name="n_levels" value="8"/>
				  <integer name="n_features_per_level" value="2"/>
				  <integer name="base_resolution" value="16"/>
				  <float name="per_level_scale" value="1.5"/>
				  <integer name="hashmap_size" value="200000"/>
				  <integer name="hidden_size" value="64"/>
				  <integer name="hidden_size" value="2"/>
				  <string name="dtype" value="float32"/>
			  </texture>
        """

    def __init__(self, props):
        super().__init__(props)

        if 'to_world' in props:
            raise RuntimeError(
                "Found a 'to_world' transformation -- not allowed. "
                "The neural texture inherits transformations from its parent shape."
            )

        self.m_encoding_type = props.get("encoding_type", "hashgrid")

        # Encoding config (consume all relevant props)
        n_levels = props.get("n_levels", 16)
        n_features_per_level = props.get("n_features_per_level", 2)
        hashmap_size = props.get("hashmap_size", 2 ** 19)
        base_resolution = props.get("base_resolution", 16)
        per_level_scale = props.get("per_level_scale", 2)
        octaves = props.get("octaves", 8)
        shift = props.get("shift", 0)
        dtype = props.get("dtype", "float32")

        if self.m_encoding_type in ("hashgrid", "permuto"):
            self.m_encoding_config = {
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "hashmap_size": hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        elif self.m_encoding_type in ("tri", "sin"):
            self.m_encoding_config = {
                "octaves": octaves,
                "shift": shift,
            }
        else:
            raise RuntimeError(f"Unknown encoding type: {self.m_encoding_type}")

        # Network config
        self.m_hidden_size = props.get("hidden_size", 0)
        self.m_num_layers = props.get("num_layers", 0)
        self.m_output_size = dr.size_v(mi.Spectrum)

        # Lazy-initialized members
        self.m_network = None
        self.m_network_weights = None
        self.m_encoding_layer = None
        self.m_encoding = None

        if dtype == "float8":
            self._initialize_network(mi.TensorXf8)
        elif dtype == "float16":
            self._initialize_network(mi.TensorXf16)
        else:
            self._initialize_network(mi.TensorXf32)

    def _initialize_network(self, dtype):
        """Initialize encoding + MLP (or linear decoder) with the given dtype."""

        # Build encoding
        if self.m_encoding_type in ("hashgrid", "permuto"):
            encoding_class = nn.HashGridEncoding if self.m_encoding_type == "hashgrid" else nn.PermutoEncoding
            self.m_encoding = encoding_class(dtype, dimension=2, **self.m_encoding_config)
            self.m_encoding_layer = nn.HashEncodingLayer(self.m_encoding)
            encoding_output_size = (
                self.m_encoding_config["n_levels"] * self.m_encoding_config["n_features_per_level"]
            )
        else:  # 'tri' / 'sin'
            encoding_class = nn.TriEncode if self.m_encoding_type == "tri" else nn.SinEncode
            self.m_encoding_layer = encoding_class(**self.m_encoding_config)
            # octaves * 2 (sin/cos) * 2D
            encoding_output_size = self.m_encoding_config["octaves"] * 2 * 2

        layers = [self.m_encoding_layer, nn.Cast(dtype)]

        if self.m_hidden_size <= 0 and self.m_num_layers <= 0:
            # direct linear decode across levels; require 3 features per level
            assert (
                self.m_encoding_type in ("hashgrid", "permuto")
                and self.m_encoding_config["n_features_per_level"] == 3
            ), "Direct decode requires hash-based encoding with 3 features/level (RGB)."
            layers.append(
                LinearDecoder(self.m_encoding_config["n_levels"], self.m_encoding_config["n_features_per_level"])
            )
        else:
            # MLP: [encoding] → hidden → RGB
            for i in range(self.m_num_layers):
                in_size = encoding_output_size if i == 0 else self.m_hidden_size
                layers.append(nn.Linear(in_size, self.m_hidden_size))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(self.m_hidden_size, self.m_output_size))
            layers.append(nn.Exp())  # keep outputs positive

        self.m_network = nn.Sequential(*layers)

        # Allocate + pack parameters for training
        rng = dr.rng(seed=0)
        self.m_network = self.m_network.alloc(dtype=dtype, size=2, rng=rng)
        self.m_network_weights, self.m_network = nn.pack(self.m_network, layout="training")

    def eval(self, si, active=True):
        """Evaluate texture color at the given surface interaction UV."""
        texture_coop = self.m_network(nn.CoopVec(si.uv))
        return mi.Spectrum(texture_coop)

    def traverse(self, cb):
        """Expose differentiable parameters to Mitsuba's parameter system."""
        super().traverse(cb)

        if self.m_network_weights is not None:
            cb.put("network_weights", self.m_network_weights, mi.ParamFlags.Differentiable)

        # Hash/permuto encoding params are separate from packed network weights
        if self.m_encoding_type in ("hashgrid", "permuto") and self.m_encoding is not None:
            cb.put("encoding_params", self.m_encoding.params, mi.ParamFlags.Differentiable)
            cb.put("n_levels", mi.Int(self.m_encoding_config["n_levels"]), mi.ParamFlags.NonDifferentiable)
            cb.put(
                "n_features_per_level",
                mi.Int(self.m_encoding_config["n_features_per_level"]),
                mi.ParamFlags.NonDifferentiable,
            )
            cb.put("hashmap_size", mi.Int(self.m_encoding_config["hashmap_size"]), mi.ParamFlags.NonDifferentiable)
            cb.put("base_resolution", mi.Int(self.m_encoding_config["base_resolution"]), mi.ParamFlags.NonDifferentiable)
            cb.put("per_level_scale", mi.Float(self.m_encoding_config["per_level_scale"]), mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys=None):
        """Notify Mitsuba when parameters are updated (marks texture dirty)."""
        super().parameters_changed(keys)

    def to_string(self):
        """Compact string representation for debugging."""
        result = "NeuralTexture[\n"
        result += f"  encoding_type = {self.m_encoding_type},\n"
        result += f"  encoding_config = {self.m_encoding_config},\n"
        result += f"  hidden_size = {self.m_hidden_size},\n"
        result += f"  num_layers = {self.m_num_layers}\n"
        result += "]"
        return result


class LinearDecoder(nn.Module):
    """
    Linearly combine per-level features into RGB.
    Used when no MLP is requested and each level provides 3 features (RGB).
    """

    def __init__(self, levels: int, features_per_level: int):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level

    def __call__(self, encoded_features: nn.CoopVec, /) -> nn.CoopVec:
        # Expect 3 channels in the encoded representation
        assert (self.features_per_level == 3), f"LinearDecoder requires 3 features per level (RGB), but got {self.features_per_level}."

        array = mi.ArrayXf(encoded_features)
        res = mi.ArrayXf(0)
        for level in range(self.levels):
            res += array[level * self.features_per_level : (level + 1) * self.features_per_level]
        return nn.CoopVec(res[0], res[1], res[2])

    def __repr__(self) -> str:
        return "LinearDecoder()"
