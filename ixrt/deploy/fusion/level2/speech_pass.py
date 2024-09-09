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

import dataclasses
from typing import List

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from ixrt.deploy.api import *
from ixrt.deploy.backend.onnx.converter import (
    convert_onnx_operator,
    default_converter,
)
from ixrt.deploy.backend.torch.executor.operators._operators import to_py_type
from ixrt.deploy.fusion import BasePass, PassSequence, PatternGraph
from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.ir import operator_attr as attr
from ixrt.deploy.ir import operator_attr as op_attrs
from ixrt.deploy.ir.operator_attr import BaseOperatorAttr
from ixrt.deploy.ir.operator_type import OperatorType as OP
from ixrt.deploy.quantizer.quant_operator.base import quant_single_input_operator

from ..level2 import ClearUnusedVariablesPass


def get_constant_input_name_of_operator(graph: Graph, operator: Operator):
    const = None
    for input in operator.inputs:
        if not graph.containe_var(input):
            continue

        if not graph.is_leaf_variable(input):
            continue

        input_var = graph.get_variable(input)
        if input_var.value is not None:
            const = input
    return const


class FuseEcapaPool(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.SHAPE,
                OP.GATHER,
                OP.CAST,
                OP.MUL,
                OP.UNSQUEEZE,
                OP.LESS,
                OP.CAST,
                OP.UNSQUEEZE,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.DIV,
            ],
            self.fuse_ecapa_pool_dynamic,
            strict=False,
        )
        self.transform.find_sequence_subgraph(
            [
                OP.MUL,
                OP.UNSQUEEZE,
                OP.LESS,
                OP.CAST,
                OP.UNSQUEEZE,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.DIV,
            ],
            self.fuse_ecapa_pool_static,
            strict=False,
        )
        return graph

    def check_reduce_sum(self, div_node, graph):
        div_previous_ops = graph.get_previous_operators(div_node)
        reduce_ops = [node for node in div_previous_ops if node.op_type == "ReduceSum"]
        if len(reduce_ops) != 2:
            return False
        return True

    def fuse_ecapa_pool_static(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        unsqueeze_node: Operator = nodes[4].operator
        mul_node: Operator = nodes[5].operator
        div_node: Operator = nodes[-1].operator
        if self.check_reduce_sum(div_node, graph) == False:
            return

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )
        inputs = [
            input for input in mul_node.inputs if input != unsqueeze_node.outputs[0]
        ]
        inputs.append(nodes[0].operator.inputs[0])
        fuse_op = self.transform.make_operator(
            op_type=OP.SPEECH_ECAPA_POOL,
            inputs=inputs,
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(fuse_op)

    def fuse_ecapa_pool_dynamic(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        cast_node: Operator = nodes[2].operator
        mul_node: Operator = nodes[3].operator
        div_node: Operator = nodes[-1].operator
        if self.check_reduce_sum(div_node, graph) == False:
            return

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )
        inputs = [nodes[0].operator.inputs[0]]
        inputs.extend(
            [input for input in mul_node.inputs if input != cast_node.outputs[0]]
        )
        fuse_op = self.transform.make_operator(
            op_type=OP.SPEECH_ECAPA_POOL,
            inputs=inputs,
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(fuse_op)


class FuseEcapaAspAttn(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.SHAPE,
                OP.GATHER,
                OP.CAST,
                OP.MUL,
                OP.UNSQUEEZE,
                OP.LESS,
                OP.CAST,
                OP.UNSQUEEZE,
                OP.REDUCE_SUM,
                OP.DIV,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.CLIP,
                OP.SQRT,
                OP.UNSQUEEZE,
                OP.TILE,
                OP.CONCAT,
            ],
            self.fuse_ecapa_asp_attn_dynamic,
            strict=False,
        )
        self.transform.find_sequence_subgraph(
            [
                OP.MUL,
                OP.UNSQUEEZE,
                OP.LESS,
                OP.CAST,
                OP.UNSQUEEZE,
                OP.REDUCE_SUM,
                OP.DIV,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.UNSQUEEZE,
                OP.SUB,
                OP.POW,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.CLIP,
                OP.SQRT,
                OP.UNSQUEEZE,
                OP.TILE,
                OP.CONCAT,
            ],
            self.fuse_ecapa_asp_attn_static,
            strict=False,
        )
        return graph

    def fuse_ecapa_asp_attn_static(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        div_node: Operator = nodes[6].operator
        mul_node: Operator = nodes[7].operator

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        inputs = [input for input in mul_node.inputs if input != div_node.outputs[0]]
        inputs.append(nodes[0].operator.inputs[0])
        fuse_op = self.transform.make_operator(
            op_type=OP.SPEECH_ECAPA_ASP_ATTN,
            inputs=inputs,
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(fuse_op)

    def fuse_ecapa_asp_attn_dynamic(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        cast_node: Operator = nodes[2].operator
        mul_node: Operator = nodes[3].operator

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        inputs = [nodes[0].operator.inputs[0]]
        inputs.extend(
            [input for input in mul_node.inputs if input != cast_node.outputs[0]]
        )
        fuse_op = self.transform.make_operator(
            op_type=OP.SPEECH_ECAPA_ASP_ATTN,
            inputs=inputs,
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(fuse_op)


class FuseEcapaScorePool(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.EQUAL,
                OP.WHERE,
                OP.SOFTMAX,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.UNSQUEEZE,
                OP.SUB,
                OP.POW,
                OP.MUL,
                OP.REDUCE_SUM,
                OP.CLIP,
                OP.SQRT,
                OP.CONCAT,
                OP.UNSQUEEZE,
            ],
            self.fuse_ecapa_score_pool,
            strict=False,
        )
        # self.transform.find_sequence_subgraph(
        #     [OP.MUL, OP.UNSQUEEZE, OP.LESS, OP.CAST, OP.UNSQUEEZE, OP.REDUCE_SUM, OP.DIV,
        #     OP.MUL, OP.REDUCE_SUM, OP.CLIP, OP.SQRT, OP.UNSQUEEZE, OP.TILE, OP.CONCAT],
        #     self.fuse_ecapa_asp_attn_static,
        #     strict=False,
        # )
        return graph

    def fuse_ecapa_score_pool(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        equal_node: Operator = nodes[0].operator
        where_node: Operator = nodes[1].operator
        softmax_node: Operator = nodes[2].operator
        mul_node: Operator = nodes[3].operator

        where_previous_nodes = graph.get_previous_operators(where_node)
        input_score = [node for node in where_previous_nodes if node != equal_node][0]
        input_data = [
            input for input in mul_node.inputs if input != softmax_node.outputs[0]
        ]
        input_data = input_data[0]

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        inputs = [input_data, input_score.outputs[0], self.transform.input_names[-1]]
        fuse_op = self.transform.make_operator(
            op_type=OP.SPEECH_ECAPA_SCORE_POOL,
            inputs=inputs,
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(fuse_op)


def EcapaTdnnPass():
    return PassSequence(
        FuseEcapaPool(),
        FuseEcapaAspAttn(),
        FuseEcapaScorePool(),
        ClearUnusedVariablesPass(),
    )
