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

from ixrt.deploy.core import Registry

from ...ir import Graph, Operator
from ...ir.operator_type import OperatorType as OP
from ..base_pass import BasePass, registe_pass

proc_getattr_registry = Registry("ProcessGetAttrRegistry")


@registe_pass(level=0)
class GetAttrPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if op.op_type == OP.GETATTR:
                self.eliminate_getattr(graph, op)

        return graph

    def eliminate_getattr(self, graph: Graph, op: Operator):
        if not proc_getattr_registry.containe(op.attributes.name):
            print(f"Warning: Cannot eliminate getattr operator, got {op}.")
            return

        return proc_getattr_registry.get(op.attributes.name)(graph, op)

    @staticmethod
    @proc_getattr_registry.registe(name="shape")
    def process_shape(graph: Graph, op: Operator):
        op.op_type = OP.SHAPE
