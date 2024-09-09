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

from typing import Callable, Tuple, Union

import torch

from ..ir.data_type import get_type_bit_width
from .quant_operator_config import QuantGrain, QuantMode, QuantPolicy
from .quant_paramter import QuantParameter


def compute_scale_zero_point(
    tensor_min, tensor_max, policy: QuantPolicy
) -> Tuple[torch.Tensor, torch.Tensor]:
    def compute_item(val_min, val_max):
        qrange = pow(2, get_type_bit_width(policy.qtype)) - 1

        if policy.mode == QuantMode.SYMMETRICAL:
            max_range = max(abs(val_min), abs(val_max))
            scale = (2 * max_range) / qrange
            scale = torch.clip(scale, 1e-7, 1000)
            return scale, scale.new_tensor(0)

        if policy.mode == QuantMode.ASYMMETRICAL:
            range = val_max - val_min
            scale = range / qrange
            zero_point = policy.qtype_min - (val_min / scale).round()
            return scale, zero_point

    if policy.grain == QuantGrain.PER_TENSOR:
        return compute_item(tensor_min, tensor_max)

    scales = []
    zero_points = []
    for val_min, val_max in zip(tensor_min, tensor_max):
        s, z = compute_item(val_min, val_max)
        scales.append(s)
        zero_points.append(z)

    return torch.stack(scales), torch.stack(zero_points)


def quantize_tensor(
    input, scale, zero_point, qmin, qmax, round: Union[str, Callable] = "nearest"
):
    qout = torch.clamp(input / scale + zero_point, qmin, qmax)
    if round == "nearest":
        return qout.round().int().float()
    if round == "floor":
        return qout.int().float()
    if round == "ceil":
        return qout.ceil().float()
    if callable(round):
        return round(qout)
    raise RuntimeError(f"Not support round {round}.")


def dequantize_tensor(input, scale, zero_point):
    return scale * (input - zero_point)


def _get_scale_zero_point(input, param: QuantParameter):
    scale = param.scale
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=input.device)

    zero_point = param.zero_point
    if not torch.is_tensor(zero_point):
        zero_point = torch.tensor(zero_point, device=input.device)

    if param.per_channel:
        qparam_shape = [1] * input.ndim
        qparam_shape[param.quant_dim] = -1
        scale = scale.reshape(*qparam_shape)
        zero_point = zero_point.reshape(*qparam_shape)

    return scale, zero_point


def quantize(
    input: torch.Tensor, param: QuantParameter, round: Union[str, Callable] = "nearest"
):
    scale, zero_point = _get_scale_zero_point(input, param)
    return quantize_tensor(
        input, scale, zero_point, param.qtype_min, param.qtype_max, round
    )


def dequantize(input: torch.Tensor, param: QuantParameter):
    scale, zero_point = _get_scale_zero_point(input, param)
    return dequantize_tensor(input, scale, zero_point)


class FakeLinearQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        scale,
        zero_point,
        qmin,
        qmax,
        round: Union[str, Callable] = "nearest",
    ):
        out = quantize_tensor(input, scale, zero_point, qmin, qmax, round)
        out = dequantize_tensor(out, scale, zero_point)
        return out

    @staticmethod
    def backward(ctx, dx):
        return dx, None, None, None, None, None


def fake_linear_quantizer(
    input: torch.Tensor, param: QuantParameter, round: Union[str, Callable] = "nearest"
):
    scale, zero_point = _get_scale_zero_point(input, param)
    return FakeLinearQuantizer.apply(
        input, scale, zero_point, param.qtype_min, param.qtype_max, round
    )
