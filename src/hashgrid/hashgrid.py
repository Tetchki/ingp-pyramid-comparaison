from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np

from util import variable


class hashgridVariable(variable.Variable):
    """Variable binding for neural hash/permuto encodings (and optional MLP).

    This class locates `*.encoding_params` and `*.network_weights` within
    `mi.traverse(scene)` keys, binds them into an `mi.ad.Optimizer`, and keeps
    them in sync across iterations.

    Expected scene parameter keys (per object or top-level):
        - "<key>.encoding_params"  or "encoding_params"  (required)
        - "<key>.network_weights"  or "network_weights"  (optional)
    """

    def __init__(
        self,
        key: str,
        initial_value: float | np.ndarray,
        **kwargs,
    ):
        super().__init__(key=key, initial_value=initial_value, **kwargs)
        self.encoding_params = None
        self.network_weights = None

        self._encoding_param_key: str | None = None
        self._network_weights_key: str | None = None

    def _extract_weights_and_encoding_parameters_from_scene_parameters(
        self, scene_parameters: mi.python.util.SceneParameters
    ) -> None:
        """Resolve parameter keys for encoding + optional network from scene."""
        preferred_enc_key = f"{self.key}.encoding_params" if self.key else "encoding_params"
        preferred_net_key = f"{self.key}.network_weights" if self.key else "network_weights"

        # Resolve encoding params
        if preferred_enc_key in scene_parameters.keys():
            enc_key = preferred_enc_key
        elif "encoding_params" in scene_parameters.keys():
            enc_key = "encoding_params"  # fallback: top-level
        else:
            raise ValueError(
                f"Missing encoding params: tried '{preferred_enc_key}' or 'encoding_params'. "
                f"Available keys: {list(scene_parameters.keys())}"
            )
        self._encoding_param_key = enc_key
        self.encoding_params = scene_parameters[enc_key]

        # Resolve network weights
        net_key = None
        if preferred_net_key in scene_parameters.keys():
            net_key = preferred_net_key
        elif "network_weights" in scene_parameters.keys():
            net_key = "network_weights"

        if net_key is not None and scene_parameters.get(net_key) is not None:
            self._network_weights_key = net_key
            self.network_weights = scene_parameters[net_key]

    def _enable_gradient(self, optimizer: mi.ad.Optimizer) -> None:
        """Ensure gradients are enabled for parameters."""
        assert self._encoding_param_key is not None and self.encoding_params is not None

        dr.enable_grad(optimizer[self._encoding_param_key])
        self.encoding_params = optimizer[self._encoding_param_key]

        if self.network_weights is not None and self._network_weights_key is not None:
            dr.enable_grad(optimizer[self._network_weights_key])
            self.network_weights = optimizer[self._network_weights_key]

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ) -> None:
        """Bind scene parameters into the optimizer and enable gradients."""
        self._extract_weights_and_encoding_parameters_from_scene_parameters(parameters)

        assert self._encoding_param_key is not None
        optimizer[self._encoding_param_key] = self.encoding_params
        if self.network_weights is not None and self._network_weights_key is not None:
            optimizer[self._network_weights_key] = self.network_weights

        self._enable_gradient(optimizer)

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ) -> None:
        """Refresh local references from optimizer each iteration."""
        _ = parameters  # kept for interface compatibility
        _ = iteration

        if self.encoding_params is None or self._encoding_param_key is None:
            raise ValueError(
                f"Encoding parameters for variable '{self.key}' have not been initialized yet!"
            )

        self.encoding_params = optimizer[self._encoding_param_key]
        if self.network_weights is not None and self._network_weights_key is not None:
            self.network_weights = optimizer[self._network_weights_key]

        self._enable_gradient(optimizer)

    def get_value(self, optimizer: mi.ad.Optimizer):
        """Return a dict of the currently bound parameter tensors."""
        if self.encoding_params is None or self._encoding_param_key is None:
            raise ValueError(
                f"Encoding parameters for variable '{self.key}' have not been initialized yet!"
            )
        values = {self._encoding_param_key: self.encoding_params}
        if self.network_weights is not None and self._network_weights_key is not None:
            values[self._network_weights_key] = self.network_weights
        return values

    def process_gradients(self, optimizer: mi.ad.Optimizer) -> None:
        """Clamp invalid grads (NaN/Inf) to zero before the optimizer step."""
        super().process_gradients(optimizer)

        if self._encoding_param_key is None:
            raise ValueError("process_gradients called before initialization (encoding_params).")

        value = optimizer[self._encoding_param_key]
        grad = dr.grad(value)
        grad = dr.select(dr.isfinite(grad), grad, 0.0)
        dr.set_grad(value, grad)

        if self.network_weights is not None and self._network_weights_key is not None:
            value_w = optimizer[self._network_weights_key]
            grad_w = dr.grad(value_w)
            grad_w = dr.select(dr.isfinite(grad_w), grad_w, 0.0)
            dr.set_grad(value_w, grad_w)