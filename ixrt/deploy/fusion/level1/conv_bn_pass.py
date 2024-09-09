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
from ixrt.deploy.ir.utils import make_variable_from_tensor

from ...ir import Graph, GraphTransform
from ...ir.operator_type import OperatorType as OP
from ..base_pass import BasePass, registe_pass
from ..matcher import GraphMatcher, PatternGraph


@registe_pass(level=1)
class ConvBnPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        pattern = PatternGraph()
        conv = pattern.build_node(OP.CONV, name="conv")
        bn = pattern.build_node(OP.BATCH_NORM, name="bn", parents=conv)

        matcher = GraphMatcher(pattern)
        matcher.findall(graph, self.fuse_conv_bn)
        return graph

    def fuse_conv_bn(self, graph: Graph, pattern_graph: PatternGraph):
        conv = pattern_graph.get_node("conv").get_operator()

        # ["input", "weight", "bias", "running_mean", "running_var"]
        bn = pattern_graph.get_node("bn").get_operator()

        def get_param(name):
            if graph.containe_var(name):
                return graph.get_variable(name).value
            return None

        new_w, new_b = self.fuse_conv_bn_weights(
            conv_w=get_param(conv.inputs[1]),
            conv_b=get_param(conv.inputs[2]) if len(conv.inputs) > 2 else None,
            bn_rm=get_param(bn.inputs[3]),
            bn_rv=get_param(bn.inputs[4]),
            bn_eps=bn.attributes.epsilon,
            bn_w=get_param(bn.inputs[1]),
            bn_b=get_param(bn.inputs[2]),
        )
        graph.get_variable(conv.inputs[1]).value = new_w

        if len(conv.inputs) > 2:
            graph.get_variable(conv.inputs[2]).value = new_b
        else:
            new_b_var = make_variable_from_tensor(
                f"{conv.name}.bias", new_b, is_parameter=True
            )
            graph.add_operator_input(conv, new_b_var)

        self.transform.delete_operator_and_link(bn, link_input=conv.outputs[0])

    def fuse_conv_bn_weights(self, conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        """Ref: https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py"""
        if conv_b is None:
            conv_b = torch.zeros_like(bn_rm)
        if bn_w is None:
            bn_w = torch.ones_like(bn_rm)
        if bn_b is None:
            bn_b = torch.zeros_like(bn_rm)
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
            [-1] + [1] * (len(conv_w.shape) - 1)
        )
        conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)
