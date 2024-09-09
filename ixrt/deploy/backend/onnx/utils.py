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
import onnx.helper
import torch
from onnx import helper
from ixrt.deploy.ir.data_type_mapping import torch_to_ir_dtype
from ixrt.deploy.ir.operator_attr import EmptyAttr
from ixrt.deploy.ir.operator_type import OperatorType
from ixrt.deploy.utils.object import flatten_container

_ONNX_CONSTANT_COUNT = 1
_ONNX_OPERATOR_COUNT = 1


def onnx_cnt_to_attr(attribute):
    new_attr = EmptyAttr()
    for attr in attribute:
        setattr(new_attr, attr.name, attr)

    return new_attr


def onnx_attr_to_dict(attribute):
    new_attr = dict()
    for attr in attribute:
        new_attr[attr.name] = helper.get_attribute_value(attr)

    return new_attr


def generate_onnx_constant_op_name(prefix="constant"):
    global _ONNX_CONSTANT_COUNT
    op_name = f"{prefix}_{_ONNX_CONSTANT_COUNT}"
    _ONNX_CONSTANT_COUNT += 1
    return op_name


def generate_onnx_operator_op_name(prefix="__intermediate_op_"):
    global _ONNX_OPERATOR_COUNT
    op_name = f"{prefix}_{_ONNX_OPERATOR_COUNT}"
    _ONNX_OPERATOR_COUNT += 1
    return op_name


def make_constant(output_name, value):
    value_key = "value"
    if isinstance(value, int):
        value_key = "value_int"
    elif isinstance(value, float):
        value_key = "value_float"
    elif isinstance(value, (list, tuple)):
        value = flatten_container(value)
        if len(value) == 0:
            value_key = "value_floats"
        elif isinstance(value[0], int):
            value_key = "value_ints"
        elif isinstance(value[0], float):
            value_key = "value_floats"
        elif isinstance(value[0], str):
            value_key = "value_strings"
    elif isinstance(value, str):
        value_key = "value_string"

    if not torch.is_tensor(value):
        value = torch.tensor(value)

    value = onnx.helper.make_tensor(
        name=output_name,
        data_type=torch_to_ir_dtype(value.dtype),
        dims=value.shape,
        vals=value.flatten().detach().cpu().numpy(),
    )

    return onnx.helper.make_node(
        OperatorType.CONSTANT, inputs=[], outputs=[output_name], **{value_key: value}
    )
