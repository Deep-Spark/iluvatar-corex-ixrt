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

__all__ = [
    "generate_variable_name",
    "generate_constant_name",
    "generate_operator_name",
    "make_variable_from_tensor",
    "eliminate_var_of_attr",
]

import torch

from .data_type_mapping import get_dtype_from_tensor
from .variable import Variable, VariableOptions


def generate_variable_name(graph, pattern="var_{idx}"):
    pattern = f"ixq_{pattern}"
    idx = -1
    while 1:
        idx += 1
        name = pattern.format(idx=idx)
        if not graph.containe_var(name):
            return name


def generate_constant_name(graph):
    return generate_variable_name(graph, "const_{idx}")


def generate_operator_name(graph, pattern="operator_{idx}"):
    pattern = f"ixq_{pattern}"
    idx = -1
    while 1:
        idx += 1
        name = pattern.format(idx=idx)
        if not graph.containe_operator(name):
            return name


def eliminate_var_of_attr(graph, attr):
    new_attr = attr.copy()
    for attr_name in new_attr.get_fields():
        attr_value = getattr(new_attr, attr_name)
        # TODO: Support recursive
        if isinstance(attr_value, Variable):
            attr_value = graph.get_var_value(attr_value)

        elif isinstance(attr_value, (tuple, list)):
            new_attr_value = []
            for _attr_val in attr_value:
                if isinstance(_attr_val, Variable):
                    _attr_val = graph.get_var_value(_attr_val)
                new_attr_value.append(_attr_val)
            attr_value = new_attr_value

        elif isinstance(attr_value, dict):
            new_attr_value = dict()
            for k, v in attr_value.items():
                if isinstance(v, Variable):
                    v = graph.get_var_value(v)
                new_attr_value[k] = v
            attr_value = new_attr_value

        setattr(new_attr, attr_name, attr_value)

    return new_attr


def make_variable_from_tensor(name, tensor, is_parameter: bool = None):
    return Variable(
        name=name,
        value=tensor,
        options=VariableOptions(
            dtype=get_dtype_from_tensor(tensor),
            shape=tensor.shape,
            is_parameter=isinstance(tensor, torch.nn.Parameter)
            if is_parameter is None
            else is_parameter,
        ),
    )
