import mitsuba as mi
import drjit as dr
import drjit.nn as nn
from mitsuba.scalar_rgb import TensorXf


class NeuralTexture(mi.Texture):
    """
    .. _emitter-neuraltexture:

    Neural texture (:monosp:`neuraltexture`)
    ----------------------------------------

    .. pluginparameters::

     * - encoding_type
       - |string|
       - Type of encoding to use. Supported types: 'sin', 'tri', 'hashgrid', 'permuto'

     * - n_levels
       - |int|
       - Number of levels in the multi-resolution encoding (for hashgrid/permuto)

     * - n_features_per_level
       - |int|
       - Number of features per level (for hashgrid/permuto)

     * - base_resolution
       - |int|
       - Base resolution for the encoding (for hashgrid/permuto)

     * - per_level_scale
       - |float|
       - Scale factor between levels (for hashgrid/permuto)

     * - hashmap_size
       - |int|
       - Size of the hash map (for hashgrid/permuto)

     * - octaves
       - |int|
       - Number of octaves (for tri/sin encodings)

     * - shift
       - |float|
       - Phase shift (for tri/sin encodings)

     * - hidden_size
       - |int|
       - Size of hidden layers in the neural network

     * - num_layers
       - |int|
       - Number of hidden layers in the neural network

    This plugin implements a neural texture that uses a neural network with
    configurable encodings to represent spatially-varying appearance on the surface
    of an arbitrary shape. The neural network maps UV coordinates to texture
    values using various encoding schemes like hash grids, triangular encodings,
    or sinusoidal encodings.

    Unlike traditional bitmap textures, this texture uses a
    neural network representation which can provide adaptive resolution and
    differentiable emission profiles for optimization tasks.

    To create a neural texture, specify the encoding type and network configuration:

    .. tabs::
        .. code-tab:: xml

            <shape type="sphere">
                <emitter type="neuraltexture">
                    <string name="encoding_type" value="hashgrid"/>
                    <integer name="n_levels" value="8"/>
                    <integer name="n_features_per_level" value="2"/>
                    <float name="per_level_scale" value="1.5"/>
                </emitter>
            </shape>

        .. code-tab:: python

            'type': 'sphere',
            'emitter': {
                'type': 'neuraltexture',
                'encoding_type': 'hashgrid',
                'n_levels': 8,
                'n_features_per_level': 2,
                'per_level_scale': 1.5
            }
    """

    def __init__(self, props):
        super().__init__(props)

        if 'to_world' in props:
            raise RuntimeError("Found a 'to_world' transformation -- this is not allowed. "
                               "The neural texture inherits this transformation from its parent "
                               "shape.")

        self.m_encoding_type = props.get("encoding_type", "hashgrid")

        # Consume all possible encoding properties upfront
        n_levels = props.get("n_levels", 16)
        n_features_per_level = props.get("n_features_per_level", 2)
        hashmap_size = props.get("hashmap_size", 2 ** 19)
        base_resolution = props.get("base_resolution", 16)
        per_level_scale = props.get("per_level_scale", 2)
        octaves = props.get("octaves", 8)
        shift = props.get("shift", 0)

        if self.m_encoding_type == "hashgrid" or self.m_encoding_type == "permuto":
            self.m_encoding_config = {
                'n_levels': n_levels,
                'n_features_per_level': n_features_per_level,
                'hashmap_size': hashmap_size,
                'base_resolution': base_resolution,
                'per_level_scale': per_level_scale,
            }
        elif self.m_encoding_type == "tri" or self.m_encoding_type == "sin":
            self.m_encoding_config = {
                'octaves': octaves,
                'shift': shift
            }
        else:
            raise RuntimeError(f"Unknown encoding type: {self.m_encoding_type}")

        # Network configuration
        self.m_hidden_size = props.get("hidden_size", 0)
        self.m_num_layers = props.get("num_layers", 0)
        self.m_output_size = dr.size_v(mi.Spectrum)

        # Initialize network structure but don't allocate yet
        self.m_network = None
        self.m_network_weights = None
        self.m_encoding_layer = None
        self.m_encoding = None

        # Initialize network (must be half precision)
        self._initialize_network(mi.TensorXf16)

    def _initialize_network(self, dtype):
        """Initialize the neural network with the specified data type."""

        # Create encoding layer based on type
        if self.m_encoding_type in ["hashgrid", "permuto"]:
            # Both hash encodings share the same initialization pattern
            encoding_class = nn.HashGridEncoding if self.m_encoding_type == "hashgrid" else nn.PermutoEncoding
            self.m_encoding = encoding_class(dtype, dimension=2, **self.m_encoding_config)
            self.m_encoding_layer = nn.HashEncodingLayer(self.m_encoding)
            encoding_output_size = self.m_encoding_config['n_levels'] * self.m_encoding_config['n_features_per_level']

        elif self.m_encoding_type in ["tri", "sin"]:
            # Trigonometric encodings
            encoding_class = nn.TriEncode if self.m_encoding_type == "tri" else nn.SinEncode
            self.m_encoding_layer = encoding_class(**self.m_encoding_config)
            encoding_output_size = self.m_encoding_config['octaves'] * 2 * 2  # N * 2 * ndim

        # Build the full network
        layers = [self.m_encoding_layer]

        # Add Cast layer for type conversion (ensures proper dtype handling)
        layers.append(nn.Cast(dtype))

        if self.m_hidden_size <= 0 and self.m_num_layers <= 0:
            assert self.m_encoding_config[
                       'n_features_per_level'] == 3, "Direct decoding without hidden layers requires 3 features per level, one per color channel."
            layers.append(LinearDecoder(self.m_encoding_config['n_levels'], self.m_encoding_config['n_features_per_level']))
        else:
            # Add hidden layers
            for i in range(self.m_num_layers):
                input_size = encoding_output_size if i == 0 else self.m_hidden_size
                layers.append(nn.Linear(input_size, self.m_hidden_size))
                layers.append(nn.LeakyReLU())  # Add LeakyReLU after EVERY hidden layer

            # Final output layer
            layers.append(nn.Linear(self.m_hidden_size, self.m_output_size))
            layers.append(nn.Exp())  # Ensure positive values

        self.m_network = nn.Sequential(*layers)

        # Initialize random number generator for weight initialization
        rng = dr.rng(seed=0)

        # Allocate network with half precision
        self.m_network = self.m_network.alloc(dtype=dtype, size=2, rng=rng)

        # Pack the entire network (including encoding layer) for all encoding types
        # Hash grid parameters are handled separately in traverse()
        self.m_network_weights, self.m_network = nn.pack(self.m_network, layout='training')

    def eval(self, si, active=True):
        """Evaluate the texture at a surface interaction point."""
        texture_coop = self.m_network(nn.CoopVec(si.uv))

        texture = mi.Spectrum(texture_coop)
        # Clamp to avoid numerical issues
        texture = dr.clip(texture, 0, 1e3)
        return texture

    def bbox(self):
        """Return the cache's bounding box."""
        shape = self.get_shape()
        return shape.bbox()

    def traverse(self, cb):
        """Handle parameter traversal for differentiation."""
        super().traverse(cb)

        if self.m_network_weights is not None:
            cb.put("network_weights", self.m_network_weights, mi.ParamFlags.Differentiable)
        # Handle hash grid and permuto encoding parameters separately (these
        # cannot be packed with the network)
        if self.m_encoding_type in ["hashgrid", "permuto"] and self.m_encoding is not None:
            cb.put("encoding_params", self.m_encoding.params, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys=None):
        """Mark emitter as dirty when parameters change."""
        super().parameters_changed(keys)
        self.set_dirty(True)

    def to_string(self):
        """Return a string representation of the texture."""
        result = f"NeuralArea[\n"
        result += f"  encoding_type = {self.m_encoding_type},\n"
        result += f"  encoding_config = {self.m_encoding_config},\n"
        result += f"  hidden_size = {self.m_hidden_size},\n"
        result += f"  num_layers = {self.m_num_layers},\n"
        result += f"  surface_area = "
        shape = self.get_shape()
        if shape:
            result += str(shape.surface_area())
        else:
            result += "<no shape attached!>"
        result += "\n]"
        return result


class LinearDecoder(nn.Module):
    """
    A simple linear decoder that sums features across multiple levels to produce RGB output.
    This module is used when no MLP is specified in the neural texture,
    and the encoding directly outputs features that can be linearly combined.
    """

    def __init__(self, levels: int, featuers_per_level: int):
        super().__init__()
        self.levels = levels
        self.features_per_level = featuers_per_level
    def __call__(self, encoded_features: nn.CoopVec, /) -> nn.CoopVec:

        assert (dr.size_v(encoded_features) < 3), "Input to LinearDecoder must contain at least 3 elements."

        array = mi.ArrayXf(encoded_features)
        res = mi.ArrayXf(0)
        for level in range(self.levels):
            res += array[level * self.features_per_level:(level + 1) * self.features_per_level]
        return nn.CoopVec(res[0], res[1], res[2])

    def __repr__(self) -> str:
        return "LinearDecoder()"
