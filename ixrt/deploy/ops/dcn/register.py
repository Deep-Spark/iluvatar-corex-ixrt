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
from typing import List

import onnx
from ixrt.deploy.api.operator import *
from ixrt.deploy.backend.onnx.converter import default_converter
from ixrt.deploy.quantizer import QuantOperatorObserverConfig
from ixrt.deploy.quantizer.quant_operator.base import quant_activations

try:
    from dcn_v2 import dcn_v2_conv
except:
    raise RuntimeError("Please install DCN from `https://github.com/lbin/DCNv2`.")


@dataclass()
class DcnAttr(BaseOperatorAttr):
    deformable_groups: int = 1
    dilation: List[int] = (1, 1)
    padding: List[int] = (1, 1)
    stride: List[int] = (1, 1)


@registe_operator(op_type="DCNv2")
class DcnOperatorRegister(BaseOperator):
    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        return default_converter(ir_graph, onnx_graph, node, attr_cls=DcnAttr)

    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        quant_activations(
            graph,
            op,
            operator_observer_config,
            num_activations=3,
            quant_outputs=quant_outputs,
        )

        weight = graph.get_variable(op.inputs[3])
        weight.set_value_observer(operator_observer_config.weight)

        bias = graph.get_variable(op.inputs[4])
        bias.set_value_observer(operator_observer_config.bias)

    def call(
        self, executor: BaseExecutor, operator: Operator, inputs: List, attr: DcnAttr
    ):
        return dcn_v2_conv(
            inputs[0],  # input
            inputs[1],  # offset
            inputs[2],  # mask
            inputs[3],  # weight
            inputs[4],  # bias
            attr.stride,
            attr.padding,
            attr.dilation,
            attr.deformable_groups,
        )
