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

import numpy as np
import torch
from ixrt.deploy.ir.utils import make_variable_from_tensor

from ...ir import Graph, GraphTransform
from ...ir.operator_type import OperatorType as OP
from ..base_pass import BasePass, registe_pass
from ..matcher import GraphMatcher, PatternGraph, build_sequence_graph


@registe_pass(level=1)
class ConvtranposeAddPass(BasePass):
    """
    ConvTranspose
         |
      Add(bias)    =>  ConvTranspose
         |
    BatchNormalization
    """

    def process(self, graph: Graph) -> Graph:
        pattern = build_sequence_graph([OP.CONV_TRANSPOSE, OP.ADD])
        matcher = GraphMatcher(pattern, strict=False)
        self.transform = GraphTransform(graph)
        matcher.findall(graph, self.fuse_convtranpose_add)
        return graph

    def fuse_convtranpose_add(self, graph: Graph, pattern_graph: PatternGraph):
        conv_transpose = pattern_graph.nodes[0].operator
        add = pattern_graph.nodes[-1].operator
        if not self.can_fused(graph, pattern_graph):
            return

        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        b_var = make_variable_from_tensor(
            add.inputs[1],
            copy.copy(graph.get_variable(add.inputs[1]).value).reshape(-1),
            is_parameter=True,
        )
        graph.add_operator_input(conv_transpose, b_var)
        self.transform.delete_operator_and_link(
            add, link_input=conv_transpose.outputs[0]
        )

    def can_fused(self, graph: Graph, pattern_graph: PatternGraph):
        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        conv_transpose = pattern_graph.nodes[0].operator
        add = pattern_graph.nodes[1].operator

        if len(conv_transpose.inputs) > 2:
            return False

        for node in pattern_graph.nodes:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) != 1:
                return False

        weight_shape = get_param(conv_transpose.inputs[1]).shape
        if not get_param(add.inputs[1]):
            return False

        bias_shape = get_param(add.inputs[1]).shape
        if (weight_shape[1] != bias_shape[1]) and (
            weight_shape[1] != math.prod(bias_shape)
        ):
            return False

        return True
