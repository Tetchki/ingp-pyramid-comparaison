# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a multiresolution image pyramid variable."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from drjit.auto.ad import Float16
from util import variable


class InstantNGPVariable(variable.Variable):
  """Represents a variable corresponding to an Instant-NGP encoding.
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

  def _extract_weights_and_encoding_parameters_from_scene_parameters(
      self, scene_parameters: mi.python.util.SceneParameters
  ):
    if f'{self.key}.encoding_params' not in scene_parameters.keys():
        raise ValueError(
            f'{self.key} does not represent an Instant-NGP encoding in the scene'
            ' parameters! Scene parameters keys:'
            f' {list(scene_parameters.keys())}'
        )

    self.encoding_params = scene_parameters[f'{self.key}.encoding_params']
    if scene_parameters.get(f'{self.key}.network_weights') is not None:
      self.network_weights = scene_parameters[f'{self.key}.network_weights']

  def _enable_gradient(self, optimizer: mi.ad.Optimizer):
    assert self.encoding_params is not None
    dr.enable_grad(optimizer[f'{self.optimizer_key}.encoding_params'])
    self.encoding_params = optimizer[f'{self.optimizer_key}.encoding_params']
    if self.network_weights is not None:
      dr.enable_grad(optimizer[f'{self.optimizer_key}.network_weights'])
      self.network_weights = optimizer[f'{self.optimizer_key}.network_weights']


  def initialize(
      self,
      optimizer: mi.ad.Optimizer,
      parameters: mi.python.util.SceneParameters,
  ):
    self._extract_weights_and_encoding_parameters_from_scene_parameters(parameters)

    optimizer[f'{self.optimizer_key}.encoding_params'] = self.encoding_params
    if self.network_weights is not None:
      optimizer[f'{self.optimizer_key}.network_weights'] = self.network_weights

    self._enable_gradient(optimizer)

  def update(
      self,
      optimizer: mi.ad.Optimizer,
      parameters: mi.python.util.SceneParameters,
      iteration: int,
  ):
    if self.encoding_params is None:
      raise ValueError(
          f'Encoding parameters for variable {self.key} have not been'
          ' initialized yet!'
      )

    self.encoding_params = Float16(optimizer[f'{self.optimizer_key}.encoding_params'])
    if self.network_weights is not None:
      self.network_weights = Float16(optimizer[f'{self.optimizer_key}.network_weights'])

    self._enable_gradient(optimizer)

  def get_value(self, optimizer: mi.ad.Optimizer):
    if self.encoding_params is None:
      raise ValueError(
          f'Encoding parameters for variable {self.key} have not been'
          ' initialized yet!'
      )
    values = {}
    values[f"{self.key}.encoding_params"] = self.encoding_params
    if self.network_weights is not None:
      values[f"{self.key}.network_weights"] = self.network_weights
    return values

  def process_gradients(self, optimizer: mi.ad.Optimizer):
    super().process_gradients(optimizer)

    value = optimizer[f'{self.optimizer_key}.encoding_params']
    gradient = dr.grad(value)
    gradient = dr.select(dr.isfinite(gradient), gradient, 0.0)
    dr.set_grad(value, gradient)

    if self.network_weights is not None:
        value = optimizer[f'{self.optimizer_key}.network_weights']
        gradient = dr.grad(value)
        gradient = dr.select(dr.isfinite(gradient), gradient, 0.0)
        dr.set_grad(value, gradient)
