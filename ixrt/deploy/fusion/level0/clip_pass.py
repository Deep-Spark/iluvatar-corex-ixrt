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

from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir.operator_type import OperatorType as OP

from ...ir import GraphTransform, generate_constant_name
from ..base_pass import BasePass, registe_pass


@registe_pass(level=0)
class FormatClip(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        for op in graph.operators.values():
            if op.op_type == OP.CLIP:
                self.format_inputs(graph, op)
        return graph

    def format_inputs(self, graph: Graph, operator: Operator):
        # 检查最小值
        if operator.num_inputs > 1 and not graph.containe_var(operator.inputs[1]):
            if operator.inputs[1] in ["", None]:
                operator.inputs[1] = generate_constant_name(graph)

            min_val = self.transform.make_variable(
                name=operator.inputs[1], value=-99999999999
            )
            self.transform.add_variable(min_val)

        # 检查最大值
        if operator.num_inputs > 2 and not graph.containe_var(operator.inputs[2]):
            if operator.inputs[2] in ["", None]:
                operator.inputs[2] = generate_constant_name(graph)

            min_val = self.transform.make_variable(
                name=operator.inputs[2], value=99999999999
            )
            self.transform.add_variable(min_val)
