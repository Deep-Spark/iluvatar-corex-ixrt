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

from typing import Dict, List, Optional, Union

from ..operator import Operator
from ..variable import Variable
from .graph_builder_mix import GraphBuilderMix
from .graph_searcher_mix import GraphSearcherMix


class GraphTransform(GraphBuilderMix, GraphSearcherMix):
    @property
    def operators(self) -> Dict[str, Operator]:
        """获取图中所有的算子"""
        return self.graph._operators

    @property
    def variables(self) -> Dict[str, Variable]:
        """获取图中的所有变量"""
        return self.graph._variables

    def containe_var(self, var: Union[str, Variable]) -> bool:
        """图中是否包含变量"""
        return self.graph.containe_var(var)

    def containe_operator(self, op: Union[str, Operator]) -> bool:
        """图中是否包含算子"""
        return self.graph.containe_operator(op)

    @property
    def inputs(self) -> Dict[str, Variable]:
        """获取图的输入"""
        return self.graph.inputs

    @property
    def input_names(self) -> List[str]:
        """获取图的输入名字"""
        return self.graph.input_names

    @property
    def outputs(self) -> Dict[str, Variable]:
        """获取图的输出"""
        return self.graph.outputs

    @property
    def output_names(self) -> List[str]:
        """获取图输出的名字"""
        return self.graph.output_names

    @property
    def quant_parameters(self) -> Dict:
        """获取图的所有量化参数"""
        return self.graph.quant_parameters

    def is_leaf_variable(self, var: Union[str, Variable]) -> bool:
        """变量是否是叶子节点"""
        return self.graph.is_leaf_variable(var)

    def is_quant_variable(self, var: Union[str, Variable]) -> bool:
        """该变量是否被量化"""
        return self.graph.is_quant_variable(var)

    def add_quant_parameter(self, name, params):
        """添加量化参数"""
        self.graph.add_quant_parameter(name, params)

    def get_quant_parameter(self, var: Union[str, Variable], **kwargs):
        """获取量化参数"""
        return self.graph.get_quant_parameter(var, **kwargs)

    def add_input(self, var: Union[str, Variable]):
        """在图中增加输入"""
        if isinstance(var, str):
            var = self.get_variable(var)
        return self.graph.add_input(var)

    def add_output(self, var: Union[str, Variable, Operator]):
        """在图中增加输出"""
        if isinstance(var, Operator):
            for out in var.outputs:
                self.graph.add_output(out)
        else:
            return self.graph.add_output(var)

    def delete_output(self, name: str):
        """在图中删除输出"""
        if name in self.graph._outputs:
            del self.graph._outputs[name]

    def replace_output(self, dest: str, src: str):
        """将图中输出的名字 src 修改为 dest"""
        if src in self.graph._outputs:
            del self.graph._outputs[src]
        self.add_output(dest)

    def add_variable(self, var: Variable):
        """添加一个变量"""
        self.graph.add_variable(var)

    def delete_variable(self, var: Union[str, Variable]) -> Variable:
        """删除变量"""
        return self.graph.delete_variable(var)

    def get_variable(self, name: str) -> Variable:
        """获取变量"""
        return self.graph.get_variable(name)

    def rename_vaiable(
        self,
        old_name,
        new_name,
        with_variables: bool = True,
        with_operator_outputs: bool = False,
    ):
        """修改变量的名字"""
        return self.graph.rename_vaiable(
            old_name, new_name, with_variables, with_operator_outputs
        )

    def add_operator(self, op: Operator):
        """添加一个算子"""
        return self.graph.add_operator(op)

    def get_operator(self, name: str) -> Operator:
        """通过名字取获取算子"""
        return self.graph.get_operator(name)

    def delete_operator(self, op: Union[str, Operator]):
        """删除算子"""
        return self.graph.delete_operator(op)

    def get_operator_input_vars(self, op: Union[str, Operator]) -> Dict[str, Variable]:
        """获取算子的输入变量"""
        return self.graph.get_operator_input_vars(op)

    def get_operator_output_vars(self, op: Union[str, Operator]) -> Dict[str, Variable]:
        """获取算子的输出变量"""
        return self.graph.get_operator_output_var(op)

    def get_src_operator(self, var: Union[str, Variable]) -> Optional[Operator]:
        """获取产生该变量的算子，即输出是 var 的算子"""
        return self.graph.get_src_operator(var)

    def get_dst_operators(self, var: Union[str, Variable]) -> List[Operator]:
        """获取所有使用该变量的所有算子，即输入包含 var 的算子"""
        return self.graph.get_dst_operators(var)

    def get_previous_operators(self, op: Union[str, Operator]) -> List[Operator]:
        """获取该算子的所有前驱"""
        return self.graph.get_previous_operators(op)

    def get_next_operators(self, op: Union[str, Operator]) -> List[Operator]:
        """获取该算子的所有后继"""
        return self.graph.get_next_operators(op)

    def clear_unused_vars(self):
        """清除图中没有被使用的变量"""
        return self.graph.clear_unused_vars()

    def toposort(self) -> List[Operator]:
        """拓扑排序"""
        return self.graph.toposort()

    def update_op_input(self, op: Operator, old_input: str, new_input: str):
        """仅仅修改算子的输入名字"""
        for i in range(len(op.inputs)):
            if op.inputs[i] == old_input:
                op.inputs[i] = new_input

    def delete_op_with_inputs(self, op: Union[str, Operator]):
        """删除算子及其所有的输入"""
        if isinstance(op, str):
            op = self.get_operator(op)
        self.delete_operator(op)
        for input in op.inputs:
            self.delete_variable(input)

    def delete_op_with_outputs(self, op: Union[str, Operator]):
        """删除算子及其所有的输出"""
        if isinstance(op, str):
            op = self.get_operator(op)
        self.delete_operator(op)
        for out in op.outputs:
            self.delete_variable(out)

    def delete_operator_and_link(self, op: Union[str, Operator], link_input=None):
        """
        删除算子并自动连接前后的算子
        before: link_input -> cur_op -> output_var -> next_ops
        after:  link_input -> next_ops
        """
        if isinstance(op, str):
            op = self.get_operator(op)

        if (link_input is None and len(op.inputs) > 1) or len(op.outputs) > 1:
            raise RuntimeError(
                "Cannot delete the operator, "
                "because it is a multi-inputs or multi-outputs operator."
            )

        if link_input is None:
            link_input = op.inputs[0]
        output = op.outputs[0]

        next_ops = self.get_next_operators(op)
        for next_op in next_ops:
            self.update_op_input(next_op, output, link_input)

        self.delete_op_with_outputs(op)

    def delete_operators_between_op_op(
        self, from_op: Operator, to_op: Operator
    ) -> bool:
        """
        删除 from_op 到 to_op (不包含to_op) 中间的 op 和 op 的输出
        before: prev_op -> from_op ... -> to_op -> next_op
        after:  prev_op, next_op
        warning:
            1. 这里并不会去连接 prev_op 和 next_op
            2. from_op 的输入不会被删除，但输出会
            3. to_op 的输入会被删除，但输出不会
        """

        if from_op is to_op:
            return True

        revers_cand_op = []
        revers_queue = [to_op]
        while len(revers_queue) != 0:
            cand_op = revers_queue.pop(0)
            cand_pre_ops = self.get_previous_operators(cand_op)
            for _op in cand_pre_ops:
                if _op.name != from_op.name and _op.name not in revers_cand_op:
                    revers_queue.append(_op)
                    # print(f"revers_queue.append {_op.name}")
                    revers_cand_op.append(_op.name)

        cand_deleted_queue = [from_op]
        while len(cand_deleted_queue) != 0:
            cand_op = cand_deleted_queue.pop(0)
            cand_next_ops = self.get_next_operators(cand_op)
            self.delete_op_with_outputs(cand_op)
            for _op in cand_next_ops:
                if _op.name != to_op.name and _op.name in revers_cand_op:
                    cand_deleted_queue.append(_op)

        self.delete_operator(to_op)
        return True

    def delete_operators_between_var_op(self, from_var: Variable, to_op: Operator):
        """
        删除变量到算子之间的所有算子和变量
        before: prev_op -> from_var ... -> to_op -> next_op
        after:  prev_op -> from_var, next_op
        warning:
            1. 这里并不会去连接 prev_op 和 next_op
            2. from_var 不会被删除
            3. to_op 的输入会被删除，但输出不会
        """
        next_ops = self.get_dst_operators(from_var)
        for op in next_ops:
            self.delete_operators_between_op_op(op, to_op)

    def cleanup(
        self,
    ):
        """
        Removes unused nodes and tensors from the graph.
        A node or tensor is considered unused if it does not contribute to any of the graph outputs.
        """
        used_node_names, used_tensor_names = self.graph.get_used_node_names()

        input_names = self.graph.input_names
        for input_name in input_names:
            if input_name not in used_tensor_names:
                self.graph.inputs.pop(input_name)

        for node_name in list(self.graph.operators.keys()):
            if node_name not in used_node_names:
                self.delete_op_with_outputs(node_name)
