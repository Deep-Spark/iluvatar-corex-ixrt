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


class FuseYoloDecoder(BasePass):
    def __init__(self, op_types, faster_impl):
        self.op_types = op_types
        self.faster_impl = faster_impl

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.RESHAPE, OP.TRANSPOSE, OP.SIGMOID, OP.SPLIT, OP.CONCAT, OP.RESHAPE],
            self.fuse_yolo_decoder_mode1,
            strict=False,
        )
        self.transform.find_sequence_subgraph(
            [OP.RESHAPE, OP.TRANSPOSE, OP.SIGMOID, OP.SLICE, OP.CONCAT, OP.RESHAPE],
            self.fuse_yolo_decoder_mode2,
            strict=False,
        )
        return graph

    # Use Split
    def fuse_yolo_decoder_mode1(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        split_node: Operator = nodes[3].operator
        concat_node: Operator = nodes[4].operator
        split_input_shape = graph.get_variable(split_node.inputs[0]).shape

        if split_input_shape is None:
            return
        if len(split_input_shape) != 5:
            return
        num_class = split_input_shape[4] - 5

        first_layer_ops = graph.get_next_operators(split_node)
        first_layer_mul = [node for node in first_layer_ops if node.op_type == "Mul"]
        last_layer_ops = graph.get_previous_operators(concat_node)
        last_layer_mul = [node for node in last_layer_ops if node.op_type == "Mul"]

        if len(first_layer_mul) == 1:
            if len(last_layer_mul) != 1:
                return
            # Mul    Pow
            #  |      |
            # Add    Mul
            #  anchor /= 4  stride /= 2
            stride, anchor = self.get_stride_anchor_mode1(
                first_layer_mul[0], last_layer_mul[0], graph
            )

        elif len(first_layer_mul) == 2:
            if len(last_layer_mul) != 2:
                return

            # Mul    Mul
            #  |      |
            # Add    Pow
            #  |      |
            # Mul    Mul
            stride, anchor = self.get_stride_anchor_mode2(last_layer_mul, graph)
        else:
            return

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        decoder_op = self.transform.make_operator(
            op_type=self.op_types,
            inputs=[nodes[0].operator.inputs[0]],
            outputs=[nodes[-1].operator.outputs[0]],
            anchor=anchor,
            num_class=num_class,
            stride=stride,
            faster_impl=self.faster_impl,
            attr_cls=op_attrs.YoloDecoderAttr,
        )
        self.transform.add_operator(decoder_op)

    # Use Slice
    def fuse_yolo_decoder_mode2(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        sigmoid_node: Operator = nodes[2].operator
        slice_node: Operator = nodes[3].operator
        concat_node: Operator = nodes[4].operator
        concat_output_shape = graph.get_variable(concat_node.outputs[0]).shape
        if concat_output_shape is None:
            return
        if len(concat_output_shape) != 5:
            return
        num_class = concat_output_shape[4] - 5

        slice_nodes = graph.get_next_operators(sigmoid_node)
        slice_nodes = [node for node in slice_nodes if node != slice_node]

        first_layer_ops = [graph.get_next_operators(node)[0] for node in slice_nodes]
        first_layer_mul = [node for node in first_layer_ops if node.op_type == "Mul"]
        last_layer_ops = graph.get_previous_operators(concat_node)
        last_layer_mul = [node for node in last_layer_ops if node.op_type == "Mul"]

        if len(first_layer_mul) == 1:
            if len(last_layer_mul) != 1:
                return

            # Mul    Pow
            #  |      |
            # Add    Mul
            #  anchor /= 4  stride /= 2
            stride, anchor = self.get_stride_anchor_mode1(
                first_layer_mul[0], last_layer_mul[0], graph
            )

        elif len(first_layer_mul) == 2:
            if len(last_layer_mul) != 2:
                return

            # Mul    Mul
            #  |      |
            # Add    Pow
            #  |      |
            # Mul    Mul
            stride, anchor = self.get_stride_anchor_mode2(last_layer_mul, graph)
        else:
            return

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        decoder_op = self.transform.make_operator(
            op_type=self.op_types,
            inputs=[nodes[0].operator.inputs[0]],
            outputs=[nodes[-1].operator.outputs[0]],
            anchor=anchor,
            num_class=num_class,
            stride=stride,
            faster_impl=self.faster_impl,
            attr_cls=op_attrs.YoloDecoderAttr,
        )
        self.transform.add_operator(decoder_op)

    def get_stride_anchor_mode2(self, mul_nodes, graph):
        v0_name = get_constant_input_name_of_operator(graph, mul_nodes[0])
        v1_name = get_constant_input_name_of_operator(graph, mul_nodes[1])
        v0 = to_py_type(graph.get_variable(v0_name).value)
        v1 = to_py_type(graph.get_variable(v1_name).value)
        if isinstance(v0, float):
            stride = v0
            anchor = v1
        else:
            stride = v1
            anchor = v0
        anchor = np.array(anchor)
        anchor = anchor[0, :, 0, 0, :] if len(anchor.shape) == 5 else anchor[:, 0, 0, :]
        anchor = list(anchor.flatten())
        return stride, anchor

    def get_stride_anchor_mode1(self, stride_node, anchor_node, graph):
        stride_node_name = get_constant_input_name_of_operator(graph, stride_node)
        anchor_node_name = get_constant_input_name_of_operator(graph, anchor_node)
        stride = to_py_type(graph.get_variable(stride_node_name).value) / 2
        anchor = to_py_type(graph.get_variable(anchor_node_name).value)
        anchor = np.array(anchor)
        anchor = anchor[0, :, 0, 0, :] if len(anchor.shape) == 5 else anchor[:, 0, 0, :]
        anchor /= 4
        anchor = list(anchor.flatten())
        return stride, anchor


def Yolov5Pass(faster_impl=0):
    return PassSequence(
        FuseYoloDecoder(OP.YOLOV5_DECODER, faster_impl),
        ClearUnusedVariablesPass(),
    )


def Yolov7Pass(faster_impl=0):
    return PassSequence(
        FuseYoloDecoder(OP.YOLOV7_DECODER, faster_impl),
        ClearUnusedVariablesPass(),
    )
