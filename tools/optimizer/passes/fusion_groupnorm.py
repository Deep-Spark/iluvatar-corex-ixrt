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
from typing import Dict

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionGroupNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "GroupNormalization", "Reshape")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Group Normalization subgraph into one node GroupNormalization:
                                  
          [Root] --> Reshape -->  InstanceNormalization  --> Reshape      --> Mul   --> Add
                                                            (root shape)    (C,1,1)   (C,1,1)
        
        """
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0:
            return

        root_input = node.input[0]

        if children[0].op_type != "InstanceNormalization":
            return

        reshape_node = None
        reshape_node = self.model.find_first_child_by_type(
            children[0], "Reshape", input_name_to_nodes, recursive=False
        )
        if reshape_node is None:
            return

        instancenorm_node = children[0]
        groups = self.model.get_initializer(instancenorm_node.input[1],True).shape[0]
        for attr in instancenorm_node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        mul_node = input_name_to_nodes[reshape_node.output[0]][0]
        
        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != "Add":
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend([reshape_node])
        subgraph_nodes.extend([last_add_node, mul_node])
        
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            last_add_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug(f"It is not safe to fuse GroupNormalization node. Skip")
            return

        weight_input = mul_node.input[
            1 - self.model.input_index(reshape_node.output[0], mul_node)
        ]
        if not self.model.is_constant_with_specified_dimension(
            weight_input, 3, "groupnorm weight"
        ):
            return
        weight = self.model.get_initializer(weight_input)
        if numpy_helper.to_array(weight).shape[0] == 1:
            return
        new_weight = numpy_helper.from_array(np.squeeze(numpy_helper.to_array(weight)), weight_input)
        weight.CopyFrom(new_weight)        

        bias_input = last_add_node.input[
            1 - self.model.input_index(mul_node.output[0], last_add_node)
        ]
        if not self.model.is_constant_with_specified_dimension(
            bias_input, 3, "groupnorm bias"
        ):
            return
        bias = self.model.get_initializer(bias_input)
        if numpy_helper.to_array(bias).shape[0] == 1:
            return
        new_bias = numpy_helper.from_array(np.squeeze(numpy_helper.to_array(bias)), bias_input)
        bias.CopyFrom(new_bias)

        self.nodes_to_remove.extend(subgraph_nodes)
        normalize_node = helper.make_node(
            "GroupNormalization",
            inputs=[node.input[0], weight_input, bias_input],
            outputs=[last_add_node.output[0]],
            name=self.model.create_node_name(
                "GroupNormalization", name_prefix="GroupNorm"
            ),
            num_groups=groups,
            epsilon=eps,
        )
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

