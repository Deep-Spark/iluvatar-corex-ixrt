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

import dataclasses
from typing import List

import numpy as np
import onnx
import torch
from ixrt.deploy.api.operator import *
from ixrt.deploy.backend.onnx.converter import default_converter
from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir import operator_attr as attr
from ixrt.deploy.ir import operator_attr as op_attrs
from ixrt.deploy.ir.operator_attr import BaseOperatorAttr
from ixrt.deploy.ir.operator_type import OperatorType as OP
from ixrt.deploy.quantizer import QuantOperatorObserverConfig
from ixrt.deploy.quantizer.quant_operator.base import quant_single_input_operator


@dataclasses.dataclass
class ReorgAttr(BaseOperatorAttr):
    stride: int = 2


@registe_operator("Reorg")
class ReorgOperator(BaseOperator):
    def call(
        self,
        executor,
        operator: Operator,
        inputs: List,
        attr: ReorgAttr,
    ):
        batch_size, num_channel, height, width = inputs[0].data.size()
        output = (
            inputs[0]
            .view(
                batch_size,
                int(num_channel / (attr.stride * attr.stride)),
                height,
                attr.stride,
                width,
                attr.stride,
            )
            .contiguous()
        )
        output = output.permute(0, 3, 5, 1, 2, 4).contiguous()
        output = output.view(
            batch_size, -1, int(height / attr.stride), int(width / attr.stride)
        )
        return output

    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        return default_converter(ir_graph, onnx_graph, node, attr_cls=ReorgAttr)

    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        return quant_single_input_operator(
            graph, op, operator_observer_config, quant_outputs=quant_outputs
        )
