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

import warnings
from typing import Union

import onnx
from ixrt.deploy.backend.onnx import onnx2ir_op
from ixrt.deploy.backend.onnx.utils import onnx_attr_to_dict
from ixrt.deploy.core import Registry
from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir.operator_attr import DynamicAttr, EmptyAttr

ONNX_OPERATORS = Registry("OnnxOperatorRegistry")


def convert_onnx_operator(op_type: Union[str, list], convert_fn=None):
    if isinstance(op_type, str):
        op_type = [op_type]

    if convert_fn is not None:
        for t in op_type:
            ONNX_OPERATORS.add_handler(t, convert_fn)
        return convert_fn

    def wrap(fn):
        for t in op_type:
            ONNX_OPERATORS.add_handler(t, fn)
        return fn

    return wrap


def get_converter(op_type, default=None):
    if ONNX_OPERATORS.containe(op_type):
        return ONNX_OPERATORS.get(op_type)

    warnings.warn(
        f"Not found converter for operator `{op_type}`, use default converter."
    )
    return default_converter
    # raise RuntimeError(
    #     f"The operator `{op_type}` need to be registed by "
    #     f"`convert_onnx_operator` or `ixrt.deploy.BaseOperator`."
    # )


def default_converter(
    ir_graph: Graph, onnx_graph: onnx.GraphProto, node, attr_cls=None
):
    if attr_cls is None:
        attr = DynamicAttr(onnx_attr_to_dict(node.attribute))
    else:
        attr = attr_cls(**onnx_attr_to_dict(node.attribute))

    return Operator(
        name=node.name,
        op_type=onnx2ir_op(node.op_type),
        inputs=list(node.input),
        outputs=list(node.output),
        attributes=attr,
    )
