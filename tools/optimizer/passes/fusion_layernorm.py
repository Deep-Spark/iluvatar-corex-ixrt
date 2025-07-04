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


class FusionLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel, hidden_size):
        self.hidden_size = hidden_size
        super().__init__(model, "LayerNormalization", "ReduceMean")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                     |                                               |
                                     +-----------------------------------------------+

         It also handles cases of duplicated sub nodes exported from older version of PyTorch:
              +----------------------+
              |                      v
              |           +-------> Sub-----------------------------------------------+
              |           |                                                           |
              |           |                                                           v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
              |                      ^
              |                      |
              +----------------------+
        """
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            return

        root_input = node.input[0]

        if children[0].op_type != "Sub" or children[0].input[0] != root_input:
            return

        if len(children) == 2:
            if children[1].op_type != "Sub" or children[1].input[0] != root_input:
                return

        div_node = None
        for child in children:
            div_node = self.model.find_first_child_by_type(
                child, "Div", input_name_to_nodes, recursive=False
            )
            if div_node is not None:
                break
        if div_node is None:
            return

        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node,
            [
                (["Sqrt", "Add", "ReduceMean", "Pow", "Sub"], [1, 0, 0, 0, 0]),
                (
                    ["Sqrt", "Add", "ReduceMean", "Pow", "Cast", "Sub"],
                    [1, 0, 0, 0, 0, 0],
                ),
            ],
            output_name_to_node,
        )
        if path_id < 0:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        second_add_node = parent_nodes[1]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0e-4:
            logger.warning(f"epsilon value is not expeced: {add_weight}")
            return

        pow_node = parent_nodes[3]
        if not self.model.find_constant_input(pow_node, 2.0) == 1:
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        is_not_have_mul_and_add = False
        is_not_have_mul_and_add_lst_node = None
        # deal with special case : layernorm do not have mul and add
        if mul_node.op_type != "Mul" and mul_node.op_type == "MatMul":
            is_not_have_mul_and_add = True
            is_not_have_mul_and_add_lst_node = div_node
        elif mul_node.op_type != "Mul":
            return

        if is_not_have_mul_and_add:
            last_add_node = is_not_have_mul_and_add_lst_node
            if self.hidden_size == 0:
                print(
                    "[Error] Please add '--hidden_size' and '--num_head' to fuse layernorm ..."
                )
                exit(0)

            subgraph_nodes = [node]
            subgraph_nodes.extend(children)
            subgraph_nodes.extend(parent_nodes[:-1])
            subgraph_nodes.extend([last_add_node])
            if len(subgraph_nodes) == 7:
                self.nodes_to_remove.extend(subgraph_nodes)
            else:
                return

            norm_name = self.model.create_node_name(
                "LayerNormalization", name_prefix="LayerNorm"
            )
            np_weights = np.ones((self.hidden_size)).astype(np.float32)
            np_weights_name = norm_name + "_weights"
            weights_tensor = helper.make_tensor(
                np_weights_name, TensorProto.FLOAT, np_weights.shape, np_weights
            )
            np_bias = np.zeros((self.hidden_size)).astype(np.float32)
            np_bias_name = norm_name + "_bias"
            bias_tensor = helper.make_tensor(
                np_bias_name, TensorProto.FLOAT, np_bias.shape, np_bias
            )
            self.model.add_initializer(weights_tensor)
            self.model.add_initializer(bias_tensor)
            normalize_node = helper.make_node(
                "LayerNormalization",
                inputs=[node.input[0], np_weights_name, np_bias_name],
                outputs=[last_add_node.output[0]],
                name=norm_name,
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("epsilon", float(add_weight))]
            )
            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
        else:
            last_add_node = input_name_to_nodes[mul_node.output[0]][0]
            if last_add_node.op_type != "Add":
                return

            subgraph_nodes = [node]
            subgraph_nodes.extend(children)
            subgraph_nodes.extend(parent_nodes[:-1])

            subgraph_nodes.extend([last_add_node, mul_node, div_node])
            if not self.model.is_safe_to_fuse_nodes(
                subgraph_nodes,
                last_add_node.output,
                input_name_to_nodes,
                output_name_to_node,
            ):
                logger.debug(f"It is not safe to fuse LayerNormalization node. Skip")
                return

            weight_input = mul_node.input[
                1 - self.model.input_index(div_node.output[0], mul_node)
            ]
            if not self.model.is_constant_with_specified_dimension(
                weight_input, 1, "layernorm weight"
            ):
                return

            bias_input = last_add_node.input[
                1 - self.model.input_index(mul_node.output[0], last_add_node)
            ]
            if not self.model.is_constant_with_specified_dimension(
                bias_input, 1, "layernorm bias"
            ):
                return

            self.nodes_to_remove.extend(subgraph_nodes)
            normalize_node = helper.make_node(
                "LayerNormalization",
                inputs=[node.input[0], weight_input, bias_input],
                outputs=[last_add_node.output[0]],
                name=self.model.create_node_name(
                    "LayerNormalization", name_prefix="LayerNorm"
                ),
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("epsilon", float(add_weight))]
            )
            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

class FusionLayerNormalizationNHWC(Fusion):
    def __init__(self, model: OnnxModel, hidden_size=1024):
        self.hidden_size = hidden_size
        super().__init__(model, "LayerNormalization", "ReduceMean")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Layer Normalization subgraph into three node transpose --> LayerNormalization --> transpose:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul   --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0) ^     (dims=3)   (dims=3)
                                     |                                                 |
                                     +-------------------------------------------------+

        """
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            return

        root_input = node.input[0]

        if children[0].op_type != "Sub" or children[0].input[0] != root_input:
            return

        div_node = None
        for child in children:
            div_node = self.model.find_first_child_by_type(
                child, "Div", input_name_to_nodes, recursive=False
            )
            if div_node is not None:
                break
        if div_node is None:
            return

        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node,
            [
                (["Sqrt", "Add", "ReduceMean", "Pow", "Sub"], [1, 0, 0, 0, 0]),
            ],
            output_name_to_node,
        )
        if path_id < 0:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        second_add_node = parent_nodes[1]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0e-4:
            logger.warning(f"epsilon value is not expeced: {add_weight}")
            return

        pow_node = parent_nodes[3]
        if not self.model.find_constant_input(pow_node, 2.0) == 1:
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        is_not_have_mul_and_add = False
        is_not_have_mul_and_add_lst_node = None
        # deal with special case : layernorm do not have mul and add
        if mul_node.op_type != "Mul" and mul_node.op_type == "MatMul":
            is_not_have_mul_and_add = True
            is_not_have_mul_and_add_lst_node = div_node
        elif mul_node.op_type != "Mul":
            return

        if is_not_have_mul_and_add:
            last_add_node = is_not_have_mul_and_add_lst_node
            if self.hidden_size == 0:
                print(
                    "[Error] Please add '--hidden_size' and '--num_head' to fuse layernorm ..."
                )
                exit(0)

            subgraph_nodes = [node]
            subgraph_nodes.extend(children)
            subgraph_nodes.extend(parent_nodes[:-1])
            subgraph_nodes.extend([last_add_node])
            if len(subgraph_nodes) == 7:
                self.nodes_to_remove.extend(subgraph_nodes)
            else:
                return

            norm_name = self.model.create_node_name(
                "LayerNormalization", name_prefix="LayerNorm"
            )
            np_weights = np.ones((self.hidden_size)).astype(np.float32)
            np_weights_name = norm_name + "_weights"
            weights_tensor = helper.make_tensor(
                np_weights_name, TensorProto.FLOAT, np_weights.shape, np_weights
            )
            np_bias = np.zeros((self.hidden_size)).astype(np.float32)
            np_bias_name = norm_name + "_bias"
            bias_tensor = helper.make_tensor(
                np_bias_name, TensorProto.FLOAT, np_bias.shape, np_bias
            )
            self.model.add_initializer(weights_tensor)
            self.model.add_initializer(bias_tensor)
            normalize_node = helper.make_node(
                "LayerNormalization",
                inputs=[node.input[0], np_weights_name, np_bias_name],
                outputs=[last_add_node.output[0]],
                name=norm_name,
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("epsilon", float(add_weight))]
            )
            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
        else:
            last_add_node = input_name_to_nodes[mul_node.output[0]][0]
            if last_add_node.op_type != "Add":
                return

            subgraph_nodes = [node]
            subgraph_nodes.extend(children)
            subgraph_nodes.extend(parent_nodes[:-1])

            subgraph_nodes.extend([last_add_node, mul_node, div_node])
            if not self.model.is_safe_to_fuse_nodes(
                subgraph_nodes,
                last_add_node.output,
                input_name_to_nodes,
                output_name_to_node,
            ):
                logger.debug(f"It is not safe to fuse LayerNormalization node. Skip")
                return

            weight_input = mul_node.input[
                1 - self.model.input_index(div_node.output[0], mul_node)
            ]
            if not self.model.is_constant_with_specified_dimension(
                weight_input, 3, "layernorm weight"
            ):
                return
            weight = self.model.get_initializer(weight_input)
            new_weight = numpy_helper.from_array(np.squeeze(numpy_helper.to_array(weight)), weight_input)
            weight.CopyFrom(new_weight)  

            bias_input = last_add_node.input[
                1 - self.model.input_index(mul_node.output[0], last_add_node)
            ]
            if not self.model.is_constant_with_specified_dimension(
                bias_input, 3, "layernorm bias"
            ):
                return
            bias = self.model.get_initializer(bias_input)
            new_bias = numpy_helper.from_array(np.squeeze(numpy_helper.to_array(bias)), bias_input)
            bias.CopyFrom(new_bias)

            self.nodes_to_remove.extend(subgraph_nodes)
            first_transpose_node_name = self.model.create_node_name(
                    "Transpose", name_prefix="Transpose"
                )
            first_transpose_node = helper.make_node(
                "Transpose",
                inputs=[node.input[0]],
                outputs=[first_transpose_node_name + "output"],
                name=first_transpose_node_name,
                perm=[0, 2, 3, 1],
            )
            normalize_node_name = self.model.create_node_name(
                    "LayerNormalization", name_prefix="LayerNorm"
            )
            normalize_node = helper.make_node(
                "LayerNormalization",
                inputs=[first_transpose_node.output[0], weight_input, bias_input],
                outputs=[normalize_node_name + "output"],
                name=normalize_node_name,
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("epsilon", float(add_weight))]
            )
            last_transpose_node_name = self.model.create_node_name(
                    "Transpose", name_prefix="Transpose"
                )
            last_transpose_node = helper.make_node(
                "Transpose",
                inputs=[normalize_node.output[0]],
                outputs=[last_add_node.output[0]],
                name=last_transpose_node_name,
                perm=[0, 3, 1, 2],
            )
            
            self.nodes_to_add.extend([first_transpose_node, normalize_node, last_transpose_node])
            self.node_name_to_graph_name[first_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
            self.node_name_to_graph_name[last_transpose_node.name] = self.this_graph_name


class FusionLayerNormalizationKeras(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model, "LayerNormalization", "GlobalAveragePool", "Keras layernorm"
        )

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
          +-------------------------------+
          |                               |
          |                               v
        [Root] -->  GlobalAveragePool-->  Sub  --> Mul --> GlobalAveragePool --> Add/Min/Max --> Sqrt --> Div --> Mul --> Add
                                           |                                                               ^
                                           |                                                               |
                                           +---------------------------------------------------------------+
        """
        children = self.model.get_children(node, input_name_to_nodes)
        # print(len(children))
        if len(children) != 1:
            return

        root_input = node.input[0]

        if children[0].op_type != "Sub" or children[0].input[0] != root_input:
            return

        div_node = None
        for child in children:
            div_node = self.model.find_first_child_by_type(
                child, "Div", input_name_to_nodes, recursive=False
            )
            if div_node is not None:
                break
        if div_node is None:
            return
        # print('div_node_name:', div_node.name)
        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node,
            [
                (
                    ["Sqrt", "Max", "Min", "Add", "GlobalAveragePool", "Mul", "Sub"],
                    [1, 0, 0, 0, None, 0, None],
                ),
            ],
            output_name_to_node,
        )
        if path_id < 0:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        second_add_node = parent_nodes[3]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0e-4:
            logger.warning(f"epsilon value is not expeced: {add_weight}")
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        if mul_node.op_type != "Mul":
            return

        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != "Add":
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend(parent_nodes[:-1])

        subgraph_nodes.extend([last_add_node, mul_node, div_node])
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            last_add_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug(f"It is not safe to fuse LayerNormalization node. Skip")
            return

        weight_input = mul_node.input[
            1 - self.model.input_index(div_node.output[0], mul_node)
        ]
        if not self.model.is_constant_with_specified_dimension(
            weight_input, 1, "layernorm weight"
        ):
            return

        bias_input = last_add_node.input[
            1 - self.model.input_index(mul_node.output[0], last_add_node)
        ]
        if not self.model.is_constant_with_specified_dimension(
            bias_input, 1, "layernorm bias"
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        normalize_node = helper.make_node(
            "LayerNormalization",
            inputs=[node.input[0], weight_input, bias_input],
            outputs=[last_add_node.output[0]],
            name=self.model.create_node_name(
                "LayerNormalization", name_prefix="LayerNorm"
            ),
        )
        normalize_node.attribute.extend(
            [helper.make_attribute("epsilon", float(add_weight))]
        )
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionLayerNormalizationTF(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "LayerNormalization", "Add", "TF")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
         Layer Norm from Tensorflow model(using keras2onnx or tf2onnx):
          +------------------------------------+
          |                                    |
          |                                    |
        (Cast_1)                               |
          |                                    |
          |                                    v                                           (B)                             (B)             (A)
         Add --> (Cast_1) --> ReduceMean -->  Sub  --> Mul --> ReduceMean --> (Cast_3) --> Add --> Sqrt --> Reciprocol --> Mul --> Mul --> Sub --> Add
          |                       |                                                                                         |       ^              ^
          |                       |                                                                                         |       |              |
          |                       +--------------------------------------------------(Cast_2)-------------------------------|-------+              |
          |                                                                                                                 v                      |
          +---------------------------------------------------------------------------------------------------------------> Mul--------------------+
        """
        return_indice = []
        _, parent_nodes, return_indice = self.model.match_parent_paths(
            node,
            [
                (
                    [
                        "Sub",
                        "Mul",
                        "Mul",
                        "Reciprocal",
                        "Sqrt",
                        "Add",
                        "ReduceMean",
                        "Mul",
                        "Sub",
                        "ReduceMean",
                    ],
                    [1, 1, None, 0, 0, 0, None, 0, 0, None],
                ),
                (
                    [
                        "Sub",
                        "Mul",
                        "Mul",
                        "Reciprocal",
                        "Sqrt",
                        "Add",
                        "Cast",
                        "ReduceMean",
                        "Mul",
                        "Sub",
                        "ReduceMean",
                    ],
                    [1, 1, None, 0, 0, 0, 0, None, 0, 0, None],
                ),
            ],
            output_name_to_node,
        )  # yapf: disable

        if parent_nodes is None:
            return

        assert len(return_indice) == 3
        if not (
            return_indice[0] in [0, 1]
            and return_indice[1] in [0, 1]
            and return_indice[2] in [0, 1]
        ):
            logger.debug(
                "return indice is exepected in [0, 1], but got {return_indice}"
            )
            return

        (
            sub_node_0,
            mul_node_0,
            mul_node_1,
            reciprocol_node,
            sqrt_node,
            add_node_0,
        ) = parent_nodes[:6]
        reduce_mean_node_0, mul_node_2, sub_node_1, reduce_mean_node_1 = parent_nodes[
            -4:
        ]

        cast_node_3 = None
        if len(parent_nodes) == 11:
            cast_node_3 = parent_nodes[6]
            assert cast_node_3.op_type == "Cast"

        mul_node_3 = self.model.match_parent(node, "Mul", 0, output_name_to_node)
        if mul_node_3 is None:
            logger.debug("mul_node_3 not found")
            return

        node_before_reduce = self.model.get_parent(
            reduce_mean_node_1, 0, output_name_to_node
        )
        root_node = (
            node_before_reduce
            if cast_node_3 is None
            else self.model.get_parent(node_before_reduce, 0, output_name_to_node)
        )
        if root_node is None:
            logger.debug("root node is none")
            return

        i, epsilon = self.model.get_constant_input(add_node_0)
        if (
            epsilon is None
            or epsilon <= 0
            or (epsilon > 1.0e-5 and cast_node_3 is None)
        ):
            logger.debug("epsilon is not matched")
            return

        if cast_node_3 is None and (
            reduce_mean_node_1.input[0] not in mul_node_3.input
            or reduce_mean_node_1.input[0] not in sub_node_1.input
        ):
            logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
            return

        if cast_node_3 is not None and (
            node_before_reduce.input[0] not in mul_node_3.input
            or reduce_mean_node_1.input[0] not in sub_node_1.input
        ):
            logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
            return

        if mul_node_2.input[0] != mul_node_2.input[1]:
            logger.debug("mul_node_2 shall have two same inputs")
            return

        subgraph_nodes = [
            node,
            sub_node_0,
            mul_node_0,
            mul_node_1,
            reciprocol_node,
            sqrt_node,
            add_node_0,
            reduce_mean_node_0,
            mul_node_2,
            sub_node_1,
            reduce_mean_node_1,
            mul_node_3,
        ]

        if cast_node_3 is not None:
            cast_node_2 = self.model.match_parent(
                mul_node_0, "Cast", 0, output_name_to_node
            )
            if cast_node_2 is None:
                logger.debug("cast_node_2 not found")
                return
            subgraph_nodes.extend([node_before_reduce, cast_node_2, cast_node_3])

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            node.output,
            self.model.input_name_to_nodes(),
            self.model.output_name_to_node(),
        ):
            logger.debug("not safe to fuse layer normalization")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        weight_input = mul_node_1.input[1]
        bias_input = sub_node_0.input[0]

        # TODO: add epsilon attribute
        fused_node = helper.make_node(
            "LayerNormalization",
            inputs=[mul_node_3.input[0], weight_input, bias_input],
            outputs=[node.output[0]],
            name=self.model.create_node_name(
                "LayerNormalization", name_prefix="LayerNorm"
            ),
        )
        fused_node.attribute.extend([helper.make_attribute("epsilon", float(epsilon))])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
