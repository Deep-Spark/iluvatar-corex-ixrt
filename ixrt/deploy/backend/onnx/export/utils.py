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

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from ixrt.deploy.ir.data_type_mapping import torch_to_ir_dtype


def convert_torch_conv_padding_to_onnx(input_dim, pads):
    if isinstance(pads, int):
        pads = [pads] * input_dim
    elif isinstance(pads, (tuple, list)):
        if input_dim == 4:
            if isinstance(pads[0], (tuple, list)):
                pads = [pads[0][0], pads[0][1], pads[1][0], pads[1][1]]
            elif len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
        elif input_dim == 5:
            if isinstance(pads[0], (tuple, list)):
                pads = [
                    pads[0][0],
                    pads[0][1],
                    pads[1][0],
                    pads[1][1],
                    pads[2][0],
                    pads[2][1],
                ]
            elif len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1], pads[0], pads[1]]
    return pads


def convert_onnx_pads_to_torch(pads: list):
    if isinstance(pads, int) or len(pads) <= 2:
        return pads

    middle = len(pads) // 2
    x_start, x_end = pads[:middle], pads[middle:]
    x_start.reverse()
    x_end.reverse()
    new_pads = []
    for start, end in zip(x_start, x_end):
        new_pads.extend([start, end])
    return new_pads


def convert_onnx_conv_padding_to_torch(x, pads):
    pads = convert_onnx_pads_to_torch(pads)
    if isinstance(pads, (int, float)):
        return x, pads

    if len(pads) == 1:
        return x, pads[0]

    if len(pads) in [2, 3]:
        if set(pads) == 1:
            return x, pads[-1]
        else:
            raise NotImplementedError()

    if len(pads) == 4:
        left, right, top, bottom = pads
        if top == bottom and left == right:
            return x, [top, left]
        x = F.pad(x, pad=pads)
        return x, 0

    if len(pads) == 6:
        left, right, top, bottom, front, back = pads
        if top == bottom and left == right and front == back:
            return x, [top, left, front]
        x = F.pad(x, pads)
        return x, 0

    raise RuntimeError(f"Not support paddding, got {pads}.")


def filter_none_attr(attr: dict) -> dict:
    new_attr = dict()
    for k, v in attr.items():
        if v is None:
            continue
        new_attr[k] = v
    return new_attr


def make_tensor(name, value):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif not torch.is_tensor(value):
        value = torch.tensor(value)

    dtype = torch.tensor(value).dtype
    dtype = torch_to_ir_dtype(dtype)

    return onnx.helper.make_tensor(name, dtype, dims=value.shape, vals=value.numpy())
