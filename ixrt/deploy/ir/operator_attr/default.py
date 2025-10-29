# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

from dataclasses import dataclass
from typing import Any, List

from .base import BaseOperatorAttr


@dataclass()
class AxisAttr(BaseOperatorAttr):
    axis: int


@dataclass()
class AxesAttr(BaseOperatorAttr):
    axes: List[int] = None


@dataclass()
class ReductionAxisAttr(BaseOperatorAttr):
    axis: int
    keepdims: int = 1


@dataclass()
class ReductionAxesAttr(BaseOperatorAttr):
    axes: List[int] = None
    keepdims: int = 1


@dataclass()
class PoolingAttr(BaseOperatorAttr):
    kernel_shape: List[int]
    auto_pad: str = "NOTSET"
    ceil_mode: int = 0
    pads: List[int] = None
    strides: List[int] = None
    count_include_pad: int = 0

@dataclass()
class BatchNormAttr(BaseOperatorAttr):
    epsilon: float
    momentum: float


@dataclass()
class CastAttr(BaseOperatorAttr):
    to: int


@dataclass()
class CeluAttr(BaseOperatorAttr):
    alpha: float = 1.0


@dataclass()
class ClipAttr(BaseOperatorAttr):
    min: float = None
    max: float = None


@dataclass()
class ConcatFromSequenceAttr(BaseOperatorAttr):
    axis: int
    new_axis: int = 0


@dataclass()
class ConvAttr(BaseOperatorAttr):
    kernel_shape: List[int]
    strides: List[int] = None
    auto_pad: str = "NOTSET"
    dilations: List[int] = None
    group: int = 1
    pads: List = None


@dataclass()
class ConvTransposeAttr(ConvAttr):
    output_padding: List[int] = None
    output_shape: List[int] = None


@dataclass()
class ConstantAttr(BaseOperatorAttr):
    sparse_value: Any = None
    value: Any = None
    value_float: float = None
    value_floats: List[float] = None
    value_int: int = None
    value_ints: List[int] = None
    value_string: str = None
    value_strings: List[str] = None


@dataclass()
class ConstantOfShape(BaseOperatorAttr):
    value: Any


@dataclass()
class ConstantOfShape(BaseOperatorAttr):
    value: Any = None


@dataclass()
class DropoutAttr(BaseOperatorAttr):
    ratio: float = 0.0


@dataclass()
class EluAttr(BaseOperatorAttr):
    alpha: float


@dataclass()
class EyeLikeAttr(BaseOperatorAttr):
    dtype: int = None
    k: int = 0


@dataclass()
class FlattenAttr(AxisAttr):
    pass


@dataclass()
class GatherNDAttr(BaseOperatorAttr):
    batch_dims: List


@dataclass()
class GemmAttr(BaseOperatorAttr):
    transA: int = None
    transB: int = None
    alpha: float = 1.0
    beta: float = 1.0
    activation: str = None


@dataclass()
class GroupNormAttr(BaseOperatorAttr):
    num_groups: int
    epsilon: float = None


@dataclass()
class GruAttr(BaseOperatorAttr):
    hidden_size: int
    activation_alpha: float = 0.01
    activation_beta: float = None
    activations: List[int] = None
    clip: float = None
    direction: str = "forward"
    linear_before_reset: Any = None


@dataclass()
class HardSigmoidAttr(BaseOperatorAttr):
    alpha: float = 0.2
    beta: float = 0.5


@dataclass()
class InstanceNormAttr(BaseOperatorAttr):
    epsilon: float


@dataclass()
class LayerNormAttr(BaseOperatorAttr):
    axis: int
    epsilon: float
    stash_type: Any = None


@dataclass()
class LeakReluAttr(BaseOperatorAttr):
    alpha: float


@dataclass()
class LrnAttr(BaseOperatorAttr):
    alpha: float
    beta: float
    size: int
    bias: Any = None


@dataclass()
class LstmAttr(BaseOperatorAttr):
    activation_alpha: float
    activation_beta: float
    hidden_size: int
    activations: List = None
    clip: float = None
    direction: str = "forward"
    input_forget: int = 1
    layout: Any = None


@dataclass()
class MaxPoolAttr(PoolingAttr):
    dilations: List[int] = None


@dataclass()
class MaxRoiPool(BaseOperatorAttr):
    pooled_shape: List[int]
    spatial_scale: float = None


@dataclass()
class MaxUnpoolAttr(BaseOperatorAttr):
    kernel_shape: List[int]
    pads: List[int] = None
    strides: List[int] = 1


@dataclass()
class PadAttr(BaseOperatorAttr):
    mode: str = "constant"

    def __post_init__(self):
        if isinstance(self.mode, bytes):
            self.mode = self.mode.decode("utf8")


@dataclass()
class RandomUniformLikeAttr(BaseOperatorAttr):
    low: float
    high: float
    dtype: int = None
    seed: int = None


@dataclass()
class DequantizeLinearAttr(BaseOperatorAttr):
    axis: int = 0
    saturate: int = 1


@dataclass()
class QuantizeLinearAttr(BaseOperatorAttr):
    axis: int = 0
    saturate: int = 1


@dataclass()
class RangeAttr(BaseOperatorAttr):
    start: float = 0
    limit: float = None
    delta: float = 1.0


@dataclass()
class ResizeAttr(BaseOperatorAttr):
    coordinate_transformation_mode: str = None
    cubic_coeff_a: float = None
    exclude_outside: int = 0
    extrapolation_value: float = 0.0
    mode: str = "nearest"
    nearest_mode: str = "round_prefer_floor"


@dataclass()
class RnnAttr(BaseOperatorAttr):
    activation_alpha: float
    activation_beta: float
    hidden_size: int
    activations: List = None
    clip: float = None
    direction: str = "forward"
    layout: Any = None


@dataclass()
class RoiAlignAttr(BaseOperatorAttr):
    coordinate_transformation_mode: str = "half_pixel"
    mode: str = "avg"
    output_height: int = 1
    output_width: int = 1
    sampling_ratio: float = 1.0
    spatial_scale: float = 1.0


@dataclass()
class ScatterElementsAttr(BaseOperatorAttr):
    axis: int
    reduction: str = "none"


@dataclass()
class SeluAttr(BaseOperatorAttr):
    alpha: float = 1.67326319217681884765625
    gamma: float = 1.05070102214813232421875


@dataclass()
class ShrinkAttr(BaseOperatorAttr):
    bias: float = 0.0
    lambd: float = 0.5


@dataclass()
class SplitAttr(BaseOperatorAttr):
    axis: int = None
    split: int = None


@dataclass()
class SplitToSequenceAttr(BaseOperatorAttr):
    axis: int
    keepdims: int = 1


@dataclass()
class TopkAttr(BaseOperatorAttr):
    axis: int
    largest: int = 1
    sorted: int = 1


@dataclass()
class TransposeAttr(BaseOperatorAttr):
    perm: List


@dataclass()
class UpsampleAttr(BaseOperatorAttr):
    mode: str = "default"


@dataclass()
class YoloDecoderAttr(BaseOperatorAttr):
    stride: float
    num_class: int
    anchor: List[float]
    faster_impl: int
