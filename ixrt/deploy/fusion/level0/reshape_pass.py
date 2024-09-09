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
from ixrt.deploy.ir import Graph
from ixrt.deploy.ir.operator_type import OperatorType as OP

from ..base_pass import BasePass, registe_pass


@registe_pass(level=0)
class FormatReshape(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if op.op_type == OP.RESHAPE:
                self.format_reshape(graph, op)

        return graph

    def format_reshape(self, graph, operator):
        shape = graph.get_variable(operator.inputs[1])
        if torch.is_tensor(shape.value):
            shape.value = shape.value.cpu().to(torch.int64)
        elif shape.value is not None:
            shape.value = torch.tensor(shape.value, dtype=torch.int64)
