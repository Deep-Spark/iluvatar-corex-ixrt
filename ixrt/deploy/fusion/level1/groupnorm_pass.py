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

from ixrt.deploy.ir.utils import make_variable_from_tensor

from ...ir import Graph, GraphTransform, Operator
from ...ir import OperatorType as OP
from ...ir import generate_operator_name
from ..base_pass import BasePass, registe_pass
from ..matcher import GraphMatcher, PatternGraph, build_sequence_graph


@registe_pass(level=1)
class FuseGroupNormPass(BasePass):
    def process(self, graph: Graph) -> Graph:

        """
                Conv                Conv
                 |
              Reshape
                 |
        InstanceNormalization   => GroupNorm
                 |
              Reshape
                 |
                Mul
                 |
                Add

        """
        pattern = build_sequence_graph(
            [OP.RESHAPE, OP.INSTANCE_NORM, OP.RESHAPE, OP.MUL, OP.ADD]
        )
        matcher = GraphMatcher(pattern, strict=False)
        self.transform = GraphTransform(graph)
        matcher.findall(graph, self.fuse_groupnorm)
        return graph

    def fuse_groupnorm(self, graph: Graph, pattern_graph: PatternGraph):
        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        reshape1 = pattern_graph.nodes[0].operator
        instance_norm = pattern_graph.nodes[1].operator
        mul = pattern_graph.nodes[-2].operator
        add = pattern_graph.nodes[-1].operator
        instance_norm_scale = graph.get_variable(instance_norm.inputs[1]).value
        scale = make_variable_from_tensor(
            mul.inputs[1],
            copy.copy(graph.get_variable(mul.inputs[1]).value).reshape(-1),
            is_parameter=True,
        )
        B = make_variable_from_tensor(
            add.inputs[1],
            copy.copy(graph.get_variable(add.inputs[1]).value).reshape(-1),
            is_parameter=True,
        )
        if not self.can_fused(graph, pattern_graph):
            return
        self.transform.delete_operators_between_op_op(reshape1, add)
        groupnorm_op = self.transform.make_operator(
            name=generate_operator_name(graph, pattern="groupnorm_{idx}"),
            op_type=OP.GROUP_NORM,
            inputs=copy.copy(reshape1.inputs[:1]),
            num_groups=instance_norm_scale.shape[0],
            num_channels=scale.shape[0],
            eps=instance_norm.attributes.epsilon,
            outputs=copy.copy(add.outputs),
        )
        groupnorm_op.is_quant_operator = (
            reshape1.is_quant_operator and add.is_quant_operator
        )

        graph.add_operator_input(groupnorm_op, scale)
        graph.add_operator_input(groupnorm_op, B)
        graph.add_operator(groupnorm_op)

    def can_fused(self, graph: Graph, pattern_graph: PatternGraph):
        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        reshape1 = pattern_graph.nodes[0].operator
        instant_norm = pattern_graph.nodes[1].operator
        reshape2 = pattern_graph.nodes[2].operator
        mul = pattern_graph.nodes[3].operator
        add = pattern_graph.nodes[-1].operator

        # 检查 各个节点(除最后一个) 的输出是不是只有一个 OP 使用
        # 如果有多个 OP 使用，则不能融合
        for node in pattern_graph.nodes[:-1]:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) != 1:
                return False
        # 检查MuL、Add 是否等于num_groups

        return True
