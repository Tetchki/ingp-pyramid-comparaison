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

"""Defines a Mip-mapped Bitmap texture."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from util import mipmap_util

if not mi.variant():
  mi.set_variant("llvm_ad_rgb")


def _flatten_index_column_major(
    index: mi.Point2u, n_rows: mi.ScalarUInt
) -> mi.UInt:
  """Flattens a 2D index in column-major order (x - rows, y - columns)."""
  return index.x * n_rows + index.y


class MipmapFlatBitmap(mi.Texture):
  """Flat Mip-mapped Bitmap texture."""

  def __init__(self, props):
    super().__init__(props)
    nested_bitmap = props["nested_bitmap"]
    self.filter_type = props.get("filter_type", "gaussian")
    self.downsampling_factor = props.get("downsampling_factor", 2)
    self.max_levels = props.get("max_levels", None)
    self.min_resolution = props.get("min_resolution", 16)
    self.volume_albedo_remap = props.get("volume_albedo_remap", False)
    self.border_mode = props.get("border_mode", "repeat")
    self.mip_bias = mi.Float(props.get("mip_bias", 0.0))
    dr.make_opaque(self.mip_bias)

    if (
        nested_bitmap.class_name() != "BitmapTexture"
        and nested_bitmap.class_name() != "BitmapTextureImpl_FP16"
        and nested_bitmap.class_name() != "BitmapTextureImpl"
    ):
      print(nested_bitmap.class_().name())
      raise ValueError(
          "MipmapBitmap only supports BitmapTexture as nested texture!"
      )

    data = mi.traverse(nested_bitmap)["data"]
    data = dr.zeros_like(data)
    nested_data = mi.TensorXf(np.array(data).astype(np.float32))

    self.base_mip_shape = mi.ScalarPoint3u(*nested_data.shape)
    dr.make_opaque(self.base_mip_shape)

    self.storage_type = None
    if nested_data.shape[-1] == 1:
      self.storage_type = mi.Float
    else:
      assert nested_data.shape[-1] == 3
      self.storage_type = mi.Color3f

    if (
        self.base_mip_shape.x % self.downsampling_factor != 0
        or self.base_mip_shape.y % self.downsampling_factor != 0
    ):
      raise ValueError(
          "The bitmap should be divisible by the downsampling factor"
          f" ({self.downsampling_factor})!"
      )

    # Generate mipmaps with gaussian filtering in a flattened Color3f Buffer
    self.mip_buffer, self.mip_buffer_offsets = (
        mipmap_util.create_flat_mip_buffer(
            nested_data,
            factor=2,
            max_levels=self.max_levels,
            min_resolution=self.min_resolution,
        )
    )
    dr.make_opaque(self.mip_buffer_offsets)

    self.available_mip_levels = dr.opaque(
        mi.UInt32, len(self.mip_buffer_offsets) - 1
    )
    self.max_mip_levels = dr.opaque(
        mi.UInt32, dr.log2i(dr.min(self.base_mip_shape.xy))
    )

  def traverse(self, callback):
    callback.put(
        "data.mip_factor",
        self.downsampling_factor,
        mi.ParamFlags.NonDifferentiable,
    )
    callback.put(
        "data.base_mip_shape",
        self.base_mip_shape,
        mi.ParamFlags.NonDifferentiable,
    )
    callback.put(
        "data.flat_buffer",
        self.mip_buffer,
        mi.ParamFlags.Differentiable,
    )
    callback.put(
        "data.flat_buffer_offsets",
        self.mip_buffer_offsets,
        mi.ParamFlags.NonDifferentiable,
    )

  def parameters_changed(self, keys: list[str]):
    self.available_mip_levels = dr.opaque(
        mi.UInt32, len(self.mip_buffer_offsets) - 1
    )
    self.max_mip_levels = dr.opaque(
        mi.UInt32, dr.log2i(dr.min(self.base_mip_shape.xy))
    )

  def _eval_mipmap(
      self,
      mip_level: mi.UInt32,
      si: mi.SurfaceInteraction3f,
      active: mi.Bool = True,
  ) -> mi.Spectrum:
    """Evaluates the mipmap at the given level."""
    if self.downsampling_factor == 2:
      # Custom speedup for factor 2.
      mip_size = self.base_mip_shape.xy >> mip_level
    else:
      level_divisor = mi.Float(self.downsampling_factor) ** mi.Float(mip_level)
      mip_size = self.base_mip_shape.xy // mi.UInt32(level_divisor)
    mip_offset = dr.gather(mi.UInt32, self.mip_buffer_offsets, mip_level)

    # Flip UVs to match Mitsuba's bitmap implementation.
    uvs = mi.Point2f(si.uv.y, si.uv.x)

    # Assuming clamp mode.
    if self.filter_type == "gaussian":
      index_float = mip_size * uvs
      shifted_index_float = index_float + 0.5

      if self.border_mode == "repeat":
        shifted_index = mi.Point2i(shifted_index_float)
        mip_size_int = mi.Point2i(mip_size)
        index_00 = (shifted_index - 1) % mip_size_int
        index_00 = mi.Point2u(
            dr.select(index_00 < 0, index_00 + mip_size, index_00)
        )
        index_11 = shifted_index % mip_size_int
        index_11 = mi.Point2u(
            dr.select(index_11 < 0, index_11 + mip_size, index_11)
        )
      else:
        shifted_index = mi.Point2u(shifted_index_float)
        index_00 = dr.clip(shifted_index, 1, mip_size) - 1  # Handles unsigned.
        index_11 = dr.clip(shifted_index, 0, mip_size - 1)
      index_10 = mi.Point2u(index_11.x, index_00.y)
      index_01 = mi.Point2u(index_00.x, index_11.y)

      weight = shifted_index_float - dr.floor(shifted_index_float)

      f00 = dr.gather(
          self.storage_type,
          self.mip_buffer,
          _flatten_index_column_major(index_00, mip_size.y) + mip_offset,
          active,
      )
      f01 = dr.gather(
          self.storage_type,
          self.mip_buffer,
          _flatten_index_column_major(index_01, mip_size.y) + mip_offset,
          active,
      )
      f10 = dr.gather(
          self.storage_type,
          self.mip_buffer,
          _flatten_index_column_major(index_10, mip_size.y) + mip_offset,
          active,
      )
      f11 = dr.gather(
          self.storage_type,
          self.mip_buffer,
          _flatten_index_column_major(index_11, mip_size.y) + mip_offset,
          active,
      )

      mip_result = dr.lerp(
          dr.lerp(f00, f10, weight.x), dr.lerp(f01, f11, weight.x), weight.y
      )
    else:
      index_00 = mi.Point2u(mip_size * uvs)
      mip_index = dr.clip(index_00, 0, mip_size - 1)
      mip_result = dr.gather(
          self.storage_type,
          self.mip_buffer,
          _flatten_index_column_major(mip_index, mip_size.y) + mip_offset,
          active,
      )

    return mip_result

  @dr.syntax
  def _eval(
      self, si: mi.SurfaceInteraction3f, active: mi.Bool = True
  ) -> mi.Spectrum:
    """Evaluates the mipmap."""
    mip_level_1, mip_level_2, mip_weight = mipmap_util.get_mip_levels(
        si, self.max_mip_levels, self.available_mip_levels, self.mip_bias
    )
    mip_eval_1 = self._eval_mipmap(mip_level_1, si, active)
    mip_eval_2 = self._eval_mipmap(mip_level_2, si, active)
    result = dr.lerp(mip_eval_1, mip_eval_2, mip_weight)

    if dr.hint(self.volume_albedo_remap, mode="scalar"):
      # effective_albedo to albedo conversion
      c = 8
      return dr.clip(
          (1 - dr.exp(-c * result)) * dr.rcp(1 - dr.exp(-c)), 0, 0.9999999
      )

    return result

  def eval(
      self, si: mi.SurfaceInteraction3f, active: mi.Bool = True
  ) -> mi.Spectrum:
    """Evaluates the mipmap."""
    return mi.Spectrum(self._eval(si, active))

  def eval_1(
      self, si: mi.SurfaceInteraction3f, active: mi.Bool = True
  ) -> mi.Float:
    """Evaluates the monochromatic mipmap."""
    if self.storage_type != mi.Float:
      # The following is is not great but roughness texture e.g. can be stored
      # as colors in mitsuba so we have to allow this.
      return self._eval(si, active).x
      # raise ValueError(
      #     "eval_1 can only be called for a monochromatic bitmap, yet storage"
      #     f" type is {self.storage_type}."
      # )
    return self._eval(si, active)

  def eval_3(
      self, si: mi.SurfaceInteraction3f, active: mi.Bool = True
  ) -> mi.Color3f:
    """Evaluates the 3-channel mipmap."""
    if self.storage_type != mi.Color3f:
      raise ValueError(
          "eval_3 can only be called for a 3-channel bitmap, yet storage type"
          f" is {self.storage_type}."
      )
    return self._eval(si, active)

  def to_string(self) -> str:
    return (
        f"MipMapTexture[\navailable_mip_levels: {self.available_mip_levels} \n"
        f"max_mip_levels:  {self.max_mip_levels}\n"
        f"base_mip_shape:  {self.base_mip_shape}\n]"
    )


def register():
  """Registers the Mipmap plugin."""
  mi.register_texture(
      "mipmap_flat", lambda props: MipmapFlatBitmap(props)  # pylint: disable=unnecessary-lambda
  )
