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
class FuseMishPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        pattern = build_sequence_graph([OP.SOFTPLUS, OP.TANH, OP.MUL])

        matcher = GraphMatcher(pattern, strict=False)
        self.transform = GraphTransform(graph)
        matcher.findall(graph, self.fuse_mish)
        return graph

    def fuse_mish(self, graph: Graph, pattern_graph: PatternGraph):
        softplus = pattern_graph.nodes[0].operator
        mul = pattern_graph.nodes[-1].operator

        if not self.can_fused(graph, pattern_graph):
            return

        self.transform.delete_operators_between_op_op(softplus, mul)

        mish_op = Operator(
            name=generate_operator_name(graph, pattern="Mish_{idx}"),
            op_type=OP.MISH,
            inputs=copy.copy(softplus.inputs),
            outputs=copy.copy(mul.outputs),
        )
        mish_op.is_quant_operator = softplus.is_quant_operator and mul.is_quant_operator
        graph.add_operator(mish_op)

    def can_fused(self, graph: Graph, pattern_graph: PatternGraph):
        softplus = pattern_graph.nodes[0].operator
        mul = pattern_graph.nodes[-1].operator

        # 检查 Softplus, tanh 的输出是不是只有一个 OP 使用
        # 如果有多个 OP 使用，则不能融合
        for node in pattern_graph.nodes[:2]:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) != 1:
                return False

        # 检查 Mul 的输入是不是和 Softplus 是同源的
        softplus_prev_op = graph.get_previous_operators(softplus)
        if len(softplus_prev_op) != 1:
            return False

        mul_prev_op = graph.get_previous_operators(mul)
        if len(mul_prev_op) != 2:
            return False

        for op in mul_prev_op:
            if op is softplus_prev_op[0]:
                return True

        return False
