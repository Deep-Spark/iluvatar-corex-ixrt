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

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper, AttributeProto

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)

def get_op_attribute(op_node, param_name):
    # 遍历节点的所有属性
    for attr in op_node.attribute:
        if attr.name == param_name:
            # 根据属性类型获取对应的值
            if attr.type == AttributeProto.FLOAT:
                return attr.f
            elif attr.type == AttributeProto.INT:
                return attr.i
            elif attr.type == AttributeProto.STRING:
                return attr.s.decode('utf-8')
            elif attr.type == AttributeProto.TENSOR:
                return attr.t  # 返回TensorProto对象
            elif attr.type == AttributeProto.GRAPH:
                return attr.g  # 返回GraphProto对象
            elif attr.type == AttributeProto.FLOATS:
                return list(attr.floats)
            elif attr.type == AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == AttributeProto.STRINGS:
                return [s.decode('utf-8') for s in attr.strings]
            elif attr.type == AttributeProto.TENSORS:
                return list(attr.tensors)
            elif attr.type == AttributeProto.GRAPHS:
                return list(attr.graphs)
            else:
                return f"Unsupported attribute type: {attr.type}"
    
    # 如果未找到指定名称的参数
    return None

class FusionLstmTranspose(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "RnnPlugin_IxRT", ["Transpose"], "rnn")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        nodes = self.model.match_parent_path(node, ["LSTM"], [0])

        if nodes is None:
            return False

        lstm = nodes[0]

        fused_node = helper.make_node(
            "RnnPlugin_IxRT",
            inputs=[lstm.input[0],lstm.input[1] ,lstm.input[2],lstm.input[3],lstm.input[5],lstm.input[6]],
            outputs=[node.output[0],lstm.output[1],lstm.output[2]],
            name=self.model.create_node_name("LSTM", "Lstm_Transpose_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("direction", 2)])
        fused_node.attribute.extend([helper.make_attribute("hidden_size", get_op_attribute(lstm, "hidden_size"))])
        fused_node.attribute.extend([helper.make_attribute("rnn_mode", 2)])
        fused_node.attribute.extend([helper.make_attribute("linear_before_reset", 1)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend([node, lstm])

class FusionLstmSqueeze(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "RnnPlugin_IxRT", ["Squeeze"], "rnn")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        nodes = self.model.match_parent_path(node, ["LSTM"], [0])

        if nodes is None:
            return False

        lstm = nodes[0]

        fused_node = helper.make_node(
            "RnnPlugin_IxRT",
            inputs=[lstm.input[0],lstm.input[1] ,lstm.input[2],lstm.input[3],lstm.input[5],lstm.input[6]],
            outputs=[node.output[0],lstm.output[1],lstm.output[2]],
            name=self.model.create_node_name("LSTM", "Lstm_Squeeze_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("direction", 1)])
        fused_node.attribute.extend([helper.make_attribute("hidden_size", get_op_attribute(lstm, "hidden_size"))])
        fused_node.attribute.extend([helper.make_attribute("rnn_mode", 2)])
        fused_node.attribute.extend([helper.make_attribute("linear_before_reset", 1)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend([node, lstm])
    
class FusionLstm(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "RnnPlugin_IxRT", ["LSTM"], "rnn")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        lstm = node

        fused_node = helper.make_node(
            "RnnPlugin_IxRT",
            inputs=[lstm.input[0],lstm.input[1] ,lstm.input[2],lstm.input[3],lstm.input[5],lstm.input[6]],
            outputs=[node.output[0],lstm.output[1],lstm.output[2]],
            name=self.model.create_node_name("LSTM", "Lstm_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("direction", 1)])
        fused_node.attribute.extend([helper.make_attribute("hidden_size", get_op_attribute(lstm, "hidden_size"))])
        fused_node.attribute.extend([helper.make_attribute("rnn_mode", 2)])
        fused_node.attribute.extend([helper.make_attribute("linear_before_reset", 1)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend([node])
    