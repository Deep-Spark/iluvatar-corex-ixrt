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

import numpy as np
import torch
from ixrt.deploy.ir.utils import make_variable_from_tensor

from ...ir import Graph, GraphTransform
from ...ir.operator_type import OperatorType as OP
from ..base_pass import BasePass, registe_pass
from ..matcher import GraphMatcher, PatternGraph


@registe_pass(level=1)
class ConvtranposeAddBNPass(BasePass):
    """
    ConvTranspose
         |
      Add(bias)    =>  ConvTranspose
         |
    BatchNormalization
    """

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        pattern = PatternGraph()
        convTranpose = pattern.build_node(OP.CONV_TRANSPOSE, name="ConvTranspose")
        add = pattern.build_node(OP.ADD, name="add", parents=convTranpose)
        bn = pattern.build_node(OP.BATCH_NORM, name="BatchNormalization", parents=add)
        matcher = GraphMatcher(pattern)
        matcher.findall(graph, self.fuse_convtranpose_add_bn)
        return graph

    def fuse_convtranpose_add_bn(self, graph: Graph, pattern_graph: PatternGraph):
        convtranpose = pattern_graph.get_node("ConvTranspose").get_operator()
        add = pattern_graph.get_node("add").get_operator()

        # ["input", "weight", "bias", "running_mean", "running_var"]
        bn = pattern_graph.get_node("BatchNormalization").get_operator()

        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        if not self.can_fused(graph, pattern_graph):
            return

        new_w, new_b = self.fuse_conv_bn_weights(
            conv_w=get_param(convtranpose.inputs[1]),
            conv_b=get_param(convtranpose.inputs[2])
            if len(convtranpose.inputs) > 2
            else get_param(add.inputs[1]),
            bn_scale=get_param(bn.inputs[1]),
            bn_B=get_param(bn.inputs[2]),
            bn_mean=get_param(bn.inputs[3]),
            bn_var=get_param(bn.inputs[4]),
            bn_eps=bn.attributes.epsilon,
        )
        graph.get_variable(convtranpose.inputs[1]).value = new_w

        if len(convtranpose.inputs) > 2:
            graph.get_variable(convtranpose.inputs[2]).value = new_b
        else:
            new_b_var = make_variable_from_tensor(
                add.inputs[1], new_b, is_parameter=True
            )
            graph.add_operator_input(convtranpose, new_b_var)

        self.transform.delete_operator_and_link(bn, link_input=convtranpose.outputs[0])
        self.transform.delete_operator(add)

    def fuse_conv_bn_weights(
        self, conv_w, conv_b, bn_scale, bn_B, bn_mean, bn_var, bn_eps
    ):
        w_update = (bn_scale / np.sqrt(bn_var + bn_eps))[
            :, None, None, None
        ] * np.transpose(
            conv_w, [1, 0, 2, 3]
        )  # broadcast
        w_update = np.transpose(w_update, [1, 0, 2, 3])
        b_update = (bn_scale / np.sqrt(bn_var + bn_eps)) * (
            conv_b.reshape(-1) - bn_mean
        ) + bn_B
        return w_update, b_update

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
        bias_shape = get_param(add.inputs[1]).shape
        if (weight_shape[1] != bias_shape[1]) and (
            weight_shape[1] != math.prod(bias_shape)
        ):
            return False
        bn = pattern_graph.nodes[2].operator
        bn_scale_shape = get_param(bn.inputs[1]).shape
        if bn_scale_shape[0] != bias_shape[1]:
            return False
        return True
