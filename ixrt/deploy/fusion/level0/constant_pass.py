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

import torch
from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir.operator_type import OperatorType as OP

from ..base_pass import BasePass, registe_pass


@registe_pass(level=0)
class ConstantPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        ops = list(graph.operators.values())
        for op in ops:
            if op.op_type == OP.CONSTANT:
                self.delete_constant(graph, op)

        return graph

    def delete_constant(self, graph: Graph, operator: Operator):
        if operator.attributes.value is None:
            return

        graph.delete_operator(operator)

        device = graph.device
        if device is None:
            device = "cpu"

        const_out = graph.get_variable(operator.outputs[0])
        const_out.value = torch.tensor(operator.attributes.value, device=device)
