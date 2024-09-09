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

from typing import List

from ...ir import Graph, Operator, Placeholder, Variable, VariableType
from ...ir.operator_type import OperatorType
from ..base_pass import BasePass, registe_pass


@registe_pass(level=0)
class MultiOutputPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        variables = list(graph.variables.values())
        for var in variables:
            if not graph.containe_var(var):
                continue

            if var.var_type == VariableType.MULTI_OUTPUTS:
                self.eliminate_gather_for_multi_outputs(graph, var)

        return graph

    def eliminate_gather_for_multi_outputs(self, graph: Graph, var: Variable):
        src_op = graph.get_src_operator(var)
        if len(src_op.outputs) > 1:
            return

        if var.value_keys is None or len(var.value_keys) == 0:
            print(
                f"Warning: Cannot merge multi-outputs, "
                f"because var.value_keys is invalid for {src_op}."
            )
            return

        dst_ops = graph.get_dst_operators(var)

        if not self._is_eliminate(graph, var, dst_ops):
            return

        new_outputs = []
        for val_key in var.value_keys:
            new_name = f"{var.name}.{val_key}"
            idx = 0
            while graph.containe_var(new_name):
                if idx == 0:
                    new_name = new_name + "_{idx}"
                new_name = new_name.format(idx=idx)
                idx += 1
            graph.add_variable(Placeholder(new_name))
            new_outputs.append(new_name)

        for dst_op in dst_ops:
            out_idx = graph.get_var_value(dst_op.inputs[1])
            new_output_name = new_outputs[out_idx]
            graph.rename_vaiable(dst_op.outputs[0], new_output_name)
            graph.operators.pop(dst_op.name)
            graph.variables.pop(dst_op.outputs[0])

        if graph.containe_var(src_op.outputs[0]):
            graph.delete_variable(src_op.outputs[0])
        src_op.replace_outputs(new_outputs)

    def _is_eliminate(self, graph: Graph, var: Variable, dst_ops: List[Operator]):
        for dst_op in dst_ops:
            if dst_op.op_type != OperatorType.GATHER:
                return False

            index = graph.get_var_value(dst_op.inputs[1])
            if index is None:
                return False

        return True
