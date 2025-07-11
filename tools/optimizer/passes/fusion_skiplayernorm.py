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

from onnx import helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipLayerNormalization(Fusion):
    """
    Fuse Add + LayerNormalization into one node: SkipLayerNormalization
    Note: This fusion does not check the input shape of Add and LayerNormalization.
    """

    def __init__(self, model: OnnxModel):
        super().__init__(
            model, "CustomSkipLayerNormPluginDynamic_IxRT", "LayerNormalization"
        )
        # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
        self.shape_infer_helper = self.model.infer_runtime_shape(
            {"batch_size": 4, "seq_len": 7}, update=True
        )

        if self.shape_infer_helper is None:
            # TODO(tianleiwu): support subgraph in shape inference or add broadcasting in SkipLayerNormalization op.
            logger.warning("symbolic shape inference disabled or failed.")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)

        # In some models there is input_ids->gather->add->LayerNorm and one of input of the
        # add node is initializer with fixed shape which should not be fused into SkipLayerNorm
        if add is None:
            return

        for add_input in add.input:
            if self.model.get_initializer(add_input) != None:
                return

        # The number of input node of add should be 2
        if len(self.model.get_parents(add)) != 2:
            return

        if self.shape_infer_helper is not None:
            if not self.shape_infer_helper.compare_shape(add.input[0], add.input[1]):
                logger.debug(
                    "skip SkipLayerNormalization fusion since shape of inputs (%s, %s) are not same",
                    add.input[0],
                    add.input[1],
                )
                return
        else:
            layernorm_weight = self.model.get_initializer(node.input[1])
            if layernorm_weight is not None:
                layernorm_weight_arr = NumpyHelper.to_array(layernorm_weight)
                hidden_size = layernorm_weight_arr.shape[0]
            else:
                logger.debug(
                    "skip SkipLayerNormalization fusion since symbolic shape inference failed"
                )
                return

        # gather_path = self.model.match_parent_path(add, ["Gather"], [None])
        # if gather_path is not None and self.model.find_graph_input(gather_path[0].input[1]) is None:
        #     if self.model.match_parent_path(gather_path[0], ["ConstantOfShape"], [1]) is None:
        #         return

        if (
            add is not None
            and add.op_type == "Add"
            and self.model.is_safe_to_fuse_nodes(
                [add, node], node.output, input_name_to_nodes, output_name_to_node
            )
        ):
            self.nodes_to_remove.extend([add, node])

            inputs = [add.input[0], add.input[1]]
            normalize_node = helper.make_node(
                "CustomSkipLayerNormPluginDynamic_IxRT",
                inputs=inputs,
                outputs=[node.output[0]],
                name=self.model.create_node_name(
                    "SkipLayerNormalization", name_prefix="SkipLayerNorm"
                ),
            )
            normalize_node.domain = "com.iluvatar"
            if self.shape_infer_helper is not None:
                hidden_size = self.shape_infer_helper.get_edge_shape(node.input[1])[-1]
            normalize_node.attribute.extend([helper.make_attribute("ld", hidden_size)])
            normalize_node.attribute.extend([helper.make_attribute("type_id", 1)])
            normalize_node.attribute.extend(
                [
                    helper.make_attribute(
                        "beta", self.model.get_initializer(node.input[2])
                    )
                ]
            )
            normalize_node.attribute.extend(
                [
                    helper.make_attribute(
                        "gamma", self.model.get_initializer(node.input[1])
                    )
                ]
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("plugin_namespace", "")]
            )
            normalize_node.attribute.extend(
                [helper.make_attribute("plugin_version", "1")]
            )

            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionBiasSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model,
            "CustomSkipLayerNormPluginDynamic_IxRT",
            "SkipLayerNormalization",
            "add bias",
        )

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        nodes = self.model.match_parent_path(
            node, ["Add", "MatMul"], [None, None], None, return_indice
        )
        if nodes is None:
            return
        assert len(return_indice) == 2
        add_input_index = return_indice[0]
        if add_input_index >= 2:
            return

        (add, matmul) = nodes

        # bias should be one dimension
        bias_index = -1
        for i, input in enumerate(add.input):
            initializer = self.model.get_initializer(input)
            if initializer is None:
                continue
            bias_index = i
            bias_weight = NumpyHelper.to_array(initializer)
            break
        if bias_weight is None:
            logger.debug(f"Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug(f"Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [node.output[0]], input_name_to_nodes, output_name_to_node
        ):
            logger.debug(
                f"Skip fusing SkipLayerNormalization with Bias since it is not safe"
            )
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            node.input[1 - add_input_index],
            matmul.output[0],
            node.input[2],
            node.input[3],
            add.input[bias_index],
        ]
        new_node = helper.make_node(
            "CustomSkipLayerNormPluginDynamic_IxRT",
            inputs=inputs,
            outputs=node.output,
            name=self.model.create_node_name(
                "SkipLayerNormalization", "SkipLayerNorm_AddBias_"
            ),
        )
        new_node.domain = "com.iluvatar"
        hidden_size = self.shape_infer_helper.get_edge_shape(node.input[2])[-1]
        new_node.attribute.extend([helper.make_attribute("ld", hidden_size)])
        new_node.attribute.extend([helper.make_attribute("type_id", 1)])
        new_node.attribute.extend(
            [helper.make_attribute("beta", self.model.get_initializer(node.input[3]))]
        )
        new_node.attribute.extend(
            [helper.make_attribute("gamma", self.model.get_initializer(node.input[2]))]
        )
        new_node.attribute.extend(
            [
                helper.make_attribute(
                    "bias", self.model.get_initializer(add.input[bias_index])
                )
            ]
        )
        new_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        new_node.attribute.extend([helper.make_attribute("plugin_version", "1")])

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
