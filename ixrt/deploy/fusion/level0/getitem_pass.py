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

from ...ir import Graph, Operator, Variable, generate_constant_name, operator_attr
from ...ir.operator_type import OperatorType as OP
from ..base_pass import BasePass, registe_pass


@registe_pass(level=0)
class GetItemPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if op.op_type == OP.GETITEM:
                self.eliminate_getitem(graph, op)

        return graph

    def eliminate_getitem(self, graph: Graph, op: Operator):
        if isinstance(op.attributes.index, slice):
            self.process_slice(graph, op)
        elif isinstance(op.attributes.index, (int, tuple, list)):
            self.convert_gather(graph, op)
        else:
            print(f"Warnning: Cannot process this operator in GetItempass, got {op}.")

    def convert_gather(self, graph: Graph, op: Operator):
        op.op_type = OP.GATHER

        index = op.attributes.index
        if not isinstance(index, Variable):
            name = generate_constant_name(graph)
            index = Variable(name, value=index)
            graph.add_variable(index)

        index = index.name

        op.inputs.append(index)
        op.attributes = operator_attr.DimsAttr(dims=0)

    def process_slice(self, graph: Graph, op: Operator):
        idx = op.attributes.index
        start = 0
        end = 9223372036854775807
        step = 1

        if idx.start is not None:
            start = idx.start

        if idx.stop is not None:
            end = idx.stop

        if idx.step is not None:
            step = idx.step

        new_attr = operator_attr.SliceAttr(starts=start, ends=end, steps=step, dims=0)
        op.op_type = OP.SLICE
        op.attributes = new_attr
