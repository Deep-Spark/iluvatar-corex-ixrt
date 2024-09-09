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

import copy

from ...ir import Graph, GraphTransform, Operator
from ...ir import OperatorType as OP
from ...ir import generate_operator_name
from ..base_pass import BasePass, registe_pass
from ..matcher import GraphMatcher, PatternGraph, build_sequence_graph


@registe_pass(level=1)
class FuseHardswishPass(BasePass):
    def process(self, graph: Graph) -> Graph:

        """
                    Conv                 Conv
                   |   \                   |
                   |   Add(3)           hardswish
                   |     \
                   |     Clip(0,6)   =>
                   \      /
                      Mul
                      |
                      Div(6)
        """

        pattern = build_sequence_graph([OP.ADD, OP.CLIP, OP.MUL, OP.DIV])
        matcher = GraphMatcher(pattern, strict=False)
        self.transform = GraphTransform(graph)
        matcher.findall(graph, self.fuse_hardswish)
        return graph

    def fuse_hardswish(self, graph: Graph, pattern_graph: PatternGraph):
        add = pattern_graph.nodes[0].operator
        div = pattern_graph.nodes[-1].operator

        if not self.can_fused(graph, pattern_graph):
            return

        self.transform.delete_operators_between_op_op(add, div)
        hardswish_op = Operator(
            name=generate_operator_name(graph, pattern="hardswish_{idx}"),
            op_type=OP.HARDSWISH,
            inputs=add.inputs[:1],
            outputs=copy.copy(div.outputs),
        )
        hardswish_op.is_quant_operator = add.is_quant_operator and div.is_quant_operator
        graph.add_operator(hardswish_op)

    def can_fused(self, graph: Graph, pattern_graph: PatternGraph):
        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        add = pattern_graph.nodes[0].operator
        clip = pattern_graph.nodes[1].operator
        mul = pattern_graph.nodes[2].operator
        div = pattern_graph.nodes[-1].operator

        # 检查 各个节点(除最后一个Div) 的输出是不是只有一个 OP 使用
        # 如果有多个 OP 使用，则不能融合
        for node in pattern_graph.nodes[:3]:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) != 1:
                return False
        add_prev_op = graph.get_previous_operators(add)
        if len(add_prev_op) != 1:
            return False

        mul_prev_op = graph.get_previous_operators(mul)
        if len(mul_prev_op) != 2:
            return False

        # 检查 Mul 的输入是不是和 add 是同源的
        if add_prev_op[0] not in mul_prev_op:
            return False
        # 检查add常量是否为3
        if get_param(add.inputs[1])[0] != 3:
            return False
        # 检查div常量是否为6
        if get_param(div.inputs[1])[0] != 6:
            return False
        return True
