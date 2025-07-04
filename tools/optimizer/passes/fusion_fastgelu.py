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
from typing import Dict, Optional

from onnx import helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionFastGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CustomGeluPluginDynamic_IxRT", "Tanh")

    def fuse(self, tanh_node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        if self.fuse_1(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_2(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_3(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_4(tanh_node, input_name_to_nodes, output_name_to_node):
            return

    def fuse_1(
        self, tanh_node, input_name_to_nodes, output_name_to_node
    ) -> Optional[bool]:
        """
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)     ^
              |                                                              |
              +------> Mul(B=0.5)--------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_tanh = children[0]

        mul_half = self.model.match_parent(
            mul_after_tanh, "Mul", None, output_name_to_node
        )
        if mul_half is None:
            return

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        root_input = mul_half.input[0 if i == 1 else 1]

        # root_node could be None when root_input is graph input
        root_node = self.model.get_parent(
            mul_half, 0 if i == 1 else 1, output_name_to_node
        )

        mul_before_tanh = self.model.match_parent(
            tanh_node, "Mul", 0, output_name_to_node
        )
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(
            mul_before_tanh, "Add", 0 if i == 1 else 1, output_name_to_node
        )
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(
            add_before_tanh,
            "Mul",
            None,
            output_name_to_node,
            exclude=[root_node] if root_node else [],
        )
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(
            mul_after_pow, "Pow", 0 if i == 1 else 1, output_name_to_node
        )
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        if pow.input[0] != root_input:
            return

        subgraph_nodes = [
            mul_after_tanh,
            mul_half,
            add_after_tanh,
            tanh_node,
            mul_before_tanh,
            add_before_tanh,
            mul_after_pow,
            pow,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_after_tanh.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "CustomGeluPluginDynamic_IxRT",
            inputs=[root_input],
            outputs=mul_after_tanh.output,
            name=self.model.create_node_name("CustomGeluPluginDynamic_IxRT"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("type_id", 1)])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_2(
        self, tanh_node, input_name_to_nodes: Dict, output_name_to_node: Dict
    ) -> Optional[bool]:
        """
        This pattern is from Tensorflow model.
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul(B=0.5)-->Mul-->
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)                  ^
              |                                                                           |
              +---------------------------------------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_half = children[0]

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        if mul_half.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_mul_half = children[0]

        root_node = self.model.get_parent(
            mul_after_mul_half,
            0 if mul_after_mul_half.input[1] == mul_half.output[0] else 1,
            output_name_to_node,
        )
        if root_node is None:
            return

        mul_before_tanh = self.model.match_parent(
            tanh_node, "Mul", 0, output_name_to_node
        )
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(
            mul_before_tanh, "Add", 0 if i == 1 else 1, output_name_to_node
        )
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(
            add_before_tanh, "Mul", None, output_name_to_node, exclude=[root_node]
        )
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(
            mul_after_pow, "Pow", 0 if i == 1 else 1, output_name_to_node
        )
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        if pow.input[0] != root_node.output[0]:
            return

        subgraph_nodes = [
            mul_after_mul_half,
            mul_half,
            add_after_tanh,
            tanh_node,
            mul_before_tanh,
            add_before_tanh,
            mul_after_pow,
            pow,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_after_mul_half.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "CustomGeluPluginDynamic_IxRT",
            inputs=[root_node.output[0]],
            outputs=mul_after_mul_half.output,
            name=self.model.create_node_name("CustomGeluPluginDynamic_IxRT"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("type_id", 1)])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_3(
        self, tanh_node, input_name_to_nodes: Dict, output_name_to_node: Dict
    ) -> Optional[bool]:
        """
        OpenAI's gelu implementation, also used in Megatron:
           Gelu(x) = x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))

        Fuse subgraph into a FastGelu node:
            +------------ Mul (B=0.79788456) -------------------+
            |                                                   |
            +-------------------------------+                   |
            |                               |                   |
            |                               v                   v
          [root] --> Mul (B=0.044715) --> Mul --> Add(B=1) --> Mul --> Tanh --> Add(B=1) --> Mul-->
            |                                                                                 ^
            |                                                                                 |
            +-----------> Mul (B=0.5) --------------------------------------------------------+
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return

        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_last = children[0]

        mul_half = self.model.match_parent(mul_last, "Mul", None, output_name_to_node)
        if mul_half is None:
            return

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        root_input = mul_half.input[0 if i == 1 else 1]

        mul_before_tanh = self.model.match_parent(
            tanh_node, "Mul", 0, output_name_to_node
        )
        if mul_before_tanh is None:
            return

        add_1 = self.model.match_parent(
            mul_before_tanh, "Add", None, output_name_to_node
        )
        if add_1 is None:
            return
        j = self.model.find_constant_input(add_1, 1.0)
        if j < 0:
            return

        mul_7978 = self.model.match_parent(
            mul_before_tanh, "Mul", None, output_name_to_node
        )
        if mul_7978 is None:
            return
        k = self.model.find_constant_input(mul_7978, 0.7978, delta=0.0001)
        if k < 0:
            return
        if mul_7978.input[0 if k == 1 else 1] != root_input:
            return

        mul_before_add_1 = self.model.match_parent(
            add_1, "Mul", 0 if j == 1 else 1, output_name_to_node
        )
        if mul_before_add_1 is None:
            return

        if mul_before_add_1.input[0] == root_input:
            another = 1
        elif mul_before_add_1.input[1] == root_input:
            another = 0
        else:
            return

        mul_0447 = self.model.match_parent(
            mul_before_add_1, "Mul", another, output_name_to_node
        )
        if mul_0447 is None:
            return
        m = self.model.find_constant_input(mul_0447, 0.0447, delta=0.0001)
        if m < 0:
            return

        if mul_0447.input[0 if m == 1 else 1] != root_input:
            return

        subgraph_nodes = [
            mul_0447,
            mul_before_add_1,
            add_1,
            mul_before_tanh,
            tanh_node,
            add_after_tanh,
            mul_7978,
            mul_half,
            mul_last,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_last.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "CustomGeluPluginDynamic_IxRT",
            inputs=[root_input],
            outputs=mul_last.output,
            name=self.model.create_node_name("CustomGeluPluginDynamic_IxRT"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("type_id", 1)])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_4(
        self, tanh_node, input_name_to_nodes: Dict, output_name_to_node: Dict
    ) -> Optional[bool]:
        """
        OpenAI's gelu implementation, also used in Megatron:
           Gelu(x) = x * 0.5 * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

        Fuse subgraph into a FastGelu node:
            +--------------------------------------------+ 
            |-----------------+                          | 
            |                 |                          |
            |---------+       |                          |
            |         |       |                          |
            |         v       v                          v 
          [root] --> Mul  --> Mul--> Mul(B=0.0447) --> Add --> Mul(B=0.79788) --> Tanh --> Add(B=1) --> Mul ---> Mul (B=0.5) -->
            |                                                                                ^
            |                                                                                |
            +--------------------------------------------------------------------------------+
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return

        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_second_last = children[0]

        children = input_name_to_nodes[mul_second_last.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return

        mul_half = children[0]

        root_input = mul_second_last.input[0]
        if add_after_tanh.output[0] == root_input:
            root_input = mul_second_last.input[1]

        mul_7978 = self.model.match_parent(
            tanh_node, "Mul", 0, output_name_to_node
        )
        if mul_7978 is None:
            return
        k = self.model.find_constant_input(mul_7978, 0.7978, delta=0.0001)
        if k < 0:
            return

        add_before_mul_7879 = self.model.match_parent(
            mul_7978, "Add", 0 if k == 1 else 1, output_name_to_node
        )
        if add_before_mul_7879 is None:
            return

        if add_before_mul_7879.input[0] == root_input:
            another = 1
        elif add_before_mul_7879.input[1] == root_input:
            another = 0
        else:
            return

        mul_0447 = self.model.match_parent(
            add_before_mul_7879, "Mul", another, output_name_to_node
        )
        if mul_0447 is None:
            return
        m = self.model.find_constant_input(mul_0447, 0.0447, delta=0.0001)
        if m < 0:
            return

        mul_before_0447 = self.model.match_parent(
            mul_0447, "Mul", 0 if m == 1 else 1, output_name_to_node
        )
        if mul_before_0447 is None:
            return

        if mul_before_0447.input[0] == root_input:
            another = 1
        elif mul_before_0447.input[1] == root_input:
            another = 0
        else:
            return

        mul_first = self.model.match_parent(
            mul_before_0447, "Mul", another, output_name_to_node
        )
        if mul_first is None:
            return

        if mul_first.input[0] != root_input:
            return
        if mul_first.input[1] != root_input:
            return

        subgraph_nodes = [
            mul_first,
            mul_before_0447,
            mul_0447,
            add_before_mul_7879,
            mul_7978,
            tanh_node,
            add_after_tanh,
            mul_second_last,
            mul_half,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_half.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "CustomGeluPluginDynamic_IxRT",
            inputs=[root_input],
            outputs=mul_half.output,
            name=self.model.create_node_name("CustomGeluPluginDynamic_IxRT"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("type_id", 1)])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True
