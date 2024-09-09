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

from onnx import helper
from ixrt.deploy.backend.onnx import ir2onnx_op
from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir.operator_type import OperatorType as OP

from .export_registry import export_onnx_operator
from .utils import make_tensor


def default_export_onnx_operator(
    graph: Graph, operator: Operator, opset_imports, onnx_graph, *args, **kwargs
):
    attrs = operator.attributes.to_dict()
    attr_names = list(attrs.keys())
    for name in attr_names:
        if attrs[name] is None:
            attrs.pop(name)

    return helper.make_node(
        op_type=ir2onnx_op(operator.op_type),
        inputs=operator.inputs,
        outputs=operator.outputs,
        name=operator.name,
        **attrs,
    )


@export_onnx_operator(OP.CONSTANT_OF_SHAPE)
def export_constant_of_shape(
    graph: Graph, operator: Operator, opset_imports, onnx_graph, *args, **kwargs
):
    attr = operator.attributes
    return helper.make_node(
        op_type=ir2onnx_op(operator.op_type),
        inputs=operator.inputs,
        outputs=operator.outputs,
        name=operator.name,
        value=make_tensor("value", attr.value),
    )
