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
from typing import List, Optional

import onnx
from onnx import GraphProto, ModelProto, TensorProto, ValueInfoProto, helper
from passes.fuse_series_bias_add import FusionSerialBiasAdd
from passes.fusion_albert_attention import FusionAlbertAttention
from passes.fusion_attention import AttentionMask, FusionAttention
from passes.fusion_biasgelu import FusionBiasGelu
from passes.fusion_customfc import (
    FusionCustomFC,
    FusionCustomFCActivation,
    FusionCustomFCGPT2,
    FusionTorchvisionVitCustomFC,
)
from passes.fusion_disentangled_attention import FusionDisentangledAttention
from passes.fusion_embedlayer import FusionEmbedLayerNormalization
from passes.fusion_fastgelu import FusionFastGelu
from passes.fusion_format_roformer import (
    FusionFormatInvalidMask,
    FusionRemoveUselessElementwise,
)
from passes.fusion_gelu import FusionGelu
from passes.fusion_gelu_approximation import FusionGeluApproximation
from passes.fusion_gpt_attention_no_past import FusionGptAttentionNoPast
from passes.fusion_layernorm import FusionLayerNormalization, FusionLayerNormalizationTF
from passes.fusion_options import FusionOptions
from passes.fusion_qordered_attention import FusionQOrderedAttention
from passes.fusion_qordered_gelu import FusionQOrderedGelu
from passes.fusion_qordered_layernorm import FusionQOrderedLayerNormalization
from passes.fusion_qordered_matmul import FusionQOrderedMatMul
from passes.fusion_reshape import FusionReshape
from passes.fusion_shape import FusionShape
from passes.fusion_skiplayernorm import (
    FusionBiasSkipLayerNormalization,
    FusionSkipLayerNormalization,
)
from passes.fusion_swinl_attention import FusionSwinLAttention
from passes.fusion_utils import FusionUtils
from passes.fusion_videobert_attention import FusionVideoBertAttention
from passes.fusion_vit_attention import FusionVITAttention, FusionTorchvisionVITAttention
from passes.fusion_xsoftmax import FusionXSoftmax
from passes.fuse_inverse_sigmoid import FusionLayerInverseSigmoid,FusionDinoLayerInverseSigmoid
from passes.fuse_l2_normalization import FusionLayerL2Normalization
from passes.fuse_omdet_attention import FusionLayerOmdetAttention
from passes.onnx_model import OnnxModel

logger = getLogger(__name__)


class BertOptimizationOptions(FusionOptions):
    """This class is deprecated"""

    def __init__(self, model_type):
        logger.warning(
            f"BertOptimizationOptions is depreciated. Please use FusionOptions instead."
        )
        super().__init__(model_type)


class BertOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BERT ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (
            num_heads > 0 and hidden_size % num_heads == 0
        )

        super().__init__(model)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionAttention(
            self, self.hidden_size, self.num_heads, self.attention_mask
        )
        self.qordered_attention_fusion = FusionQOrderedAttention(
            self, self.hidden_size, self.num_heads, self.attention_mask
        )
        self.utils = FusionUtils(self)

    def fuse_attention(self):
        self.attention_fusion.apply()
        FusionAlbertAttention(
            self, self.hidden_size, self.num_heads, self.attention_mask
        ).apply()
        FusionVideoBertAttention(self).apply()
        FusionVITAttention(self).apply()
        FusionTorchvisionVITAttention(self).apply()
        FusionSwinLAttention(self).apply()
        FusionGptAttentionNoPast(self).apply()
        # Only relevant in models with Q-DQ nodes
        self.qordered_attention_fusion.apply()

    def fuse_format_roformer(self):
        FusionRemoveUselessElementwise(self).apply()
        fusion = FusionFormatInvalidMask(self)
        fusion.apply()

    def fuse_custom_fc(self):
        fusion = FusionCustomFC(self)
        fusion.apply()

    def fuse_custom_fc_torchvision_vit(self):
        fusion = FusionTorchvisionVitCustomFC(self)
        fusion.apply()
    
    def fuse_custom_fc_activation(self):
        fusion = FusionCustomFCActivation(self)
        fusion.apply()

    def fuse_custom_fc_gpt2_classify(self):
        fusion = FusionCustomFCGPT2(self)
        fusion.apply()

    def fuse_swinT_serial_bias_add(self):
        fusion = FusionSerialBiasAdd(self)
        fusion.apply()

    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()
        fusion = FusionFastGelu(self)
        fusion.apply()
        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedGelu(self)
        fusion.apply()

    def fuse_bias_gelu(self, is_fastgelu):
        fusion = FusionBiasGelu(self, is_fastgelu)
        fusion.apply()

    def fuse_custom_xsoftmax(self):
        fusion = FusionXSoftmax(self)
        fusion.apply()

    def fuse_disentangled_attention(self):
        fusion = FusionDisentangledAttention(self)
        fusion.apply()

    def gelu_approximation(self):
        fusion = FusionGeluApproximation(self)
        fusion.apply()

    def fuse_add_bias_skip_layer_norm(self):
        fusion = FusionBiasSkipLayerNormalization(self)
        fusion.apply()

    def fuse_reshape(self):
        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_shape(self):
        fusion = FusionShape(self)
        fusion.apply()

    def fuse_embed_layer(self):
        fusion = FusionEmbedLayerNormalization(self)
        fusion.apply()

    def fuse_layer_norm(self):
        fusion = FusionLayerNormalization(self, self.hidden_size)
        fusion.apply()

        fusion = FusionLayerNormalizationTF(self)
        fusion.apply()

        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedLayerNormalization(self)
        fusion.apply()

    def fuse_skip_layer_norm(self):
        fusion = FusionSkipLayerNormalization(self)
        fusion.apply()

    # Only relevant in models with Q-DQ nodes
    def fuse_qordered_mamtul(self):
        fusion = FusionQOrderedMatMul(self)
        fusion.apply()

    def fuse_omdet_inverse_sigmoid(self):
        fusion = FusionLayerInverseSigmoid(self)
        fusion.apply()

    def fuse_dino_inverse_sigmoid(self):
        fusion = FusionDinoLayerInverseSigmoid(self)
        fusion.apply()

    def fuse_omdet_attention(self):
        fusion = FusionLayerOmdetAttention(self)
        fusion.apply()

    def fuse_l2_normalization(self):
        fusion = FusionLayerL2Normalization(self)
        fusion.apply()

    def get_graph_inputs_from_node_type(
        self, op_type: str, input_indices: List[int], casted: bool
    ):
        """
        Get graph inputs that feed into node type (like EmbedLayerNormalization or Attention).
        Returns a list of the graph input names based on the filter whether it is casted or not.
        """
        graph_inputs = []

        output_name_to_node = self.output_name_to_node()
        nodes = self.get_nodes_by_op_type(op_type)
        for node in nodes:
            bert_inputs = [node.input[i] for i in input_indices if i < len(node.input)]
            for bert_input in bert_inputs:
                if self.find_graph_input(bert_input):
                    if not casted:
                        graph_inputs.append(bert_input)
                elif bert_input in output_name_to_node:
                    parent = output_name_to_node[bert_input]
                    if (
                        parent.op_type == "Cast"
                        and self.find_graph_input(parent.input[0]) is not None
                    ):
                        if casted:
                            graph_inputs.append(parent.input[0])
        return graph_inputs

    def get_graph_inputs_from_fused_nodes(self, casted: bool):
        inputs = self.get_graph_inputs_from_node_type(
            "EmbedLayerNormalization", [0, 1, 7], casted
        )
        inputs += self.get_graph_inputs_from_node_type("Attention", [3], casted)
        return inputs

    def change_graph_input_type(
        self,
        graph: GraphProto,
        graph_input: ValueInfoProto,
        new_type: int = TensorProto.INT32,
    ):
        """Change graph input type, and add Cast node if needed.

        Args:
            graph (GraphProto): graph
            graph_input (TensorProto): input of the graph
            new_type (int, optional): new data type. Defaults to TensorProto.INT32.

        Returns:
            NodeProto: a new Cast node that added. None if Cast node is not added.
            List[NodeProto]: Cast nodes that have been removed.
        """
        assert isinstance(graph, GraphProto)
        assert isinstance(graph_input, ValueInfoProto)
        assert self.find_graph_input(graph_input.name)

        if graph_input.type.tensor_type.elem_type == int(new_type):
            return None, []

        new_cast_node = None
        nodes_to_remove = []

        input_name_to_nodes = self.input_name_to_nodes()
        if graph_input.name in input_name_to_nodes:
            nodes = input_name_to_nodes[graph_input.name]

            # For children that is not Cast node, insert a Cast node to convert int32 to original data type.
            nodes_not_cast = [node for node in nodes if node.op_type != "Cast"]
            if nodes_not_cast:
                node_name = self.create_node_name("Cast")
                output_name = node_name + "_" + graph_input.name
                new_value_info = graph.value_info.add()
                new_value_info.CopyFrom(graph_input)
                new_value_info.name = output_name
                new_cast_node = helper.make_node(
                    "Cast",
                    [graph_input.name],
                    [output_name],
                    to=int(graph_input.type.tensor_type.elem_type),
                    name=node_name,
                )
                graph.node.extend([new_cast_node])

                for node in nodes_not_cast:
                    OnnxModel.replace_node_input(node, graph_input.name, output_name)

            # For children that is Cast node, no need to insert Cast.
            # When the children is Cast to int32, we can remove that Cast node since input type is int32 now.
            nodes_cast = [node for node in nodes if node.op_type == "Cast"]
            for node in nodes_cast:
                if OnnxModel.get_node_attribute(node, "to") == int(new_type):
                    self.replace_input_of_all_nodes(node.output[0], graph_input.name)
                if not self.find_graph_output(node.output[0]):
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                self.remove_nodes(nodes_to_remove)

        graph_input.type.tensor_type.elem_type = int(new_type)
        return new_cast_node, nodes_to_remove

    def change_graph_inputs_to_int32(self):
        """Change data type of all graph inputs to int32 type, and add Cast node if needed."""
        graph = self.graph()
        add_cast_count = 0
        remove_cast_count = 0
        for graph_input in graph.input:
            new_node, removed_nodes = self.change_graph_input_type(
                graph, graph_input, TensorProto.INT32
            )
            if new_node:
                add_cast_count += 1
            remove_cast_count += len(removed_nodes)
        logger.info(
            f"Graph inputs are changed to int32. Added {add_cast_count} Cast nodes, and removed {remove_cast_count} Cast nodes."
        )

    def use_dynamic_axes(
        self, dynamic_batch_dim="batch_size", dynamic_seq_len="max_seq_len"
    ):
        """
        Update input and output shape to use dynamic axes.
        """
        bert_graph_inputs = self.get_graph_inputs_from_fused_nodes(
            casted=True
        ) + self.get_graph_inputs_from_fused_nodes(casted=False)

        dynamic_batch_inputs = {}
        for input in self.model.graph.input:
            if input.name in bert_graph_inputs:
                dim_proto = input.type.tensor_type.shape.dim[0]
                dim_proto.dim_param = dynamic_batch_dim
                if dynamic_seq_len is not None:
                    dim_proto = input.type.tensor_type.shape.dim[1]
                    dim_proto.dim_param = dynamic_seq_len

        for output in self.model.graph.output:
            dim_proto = output.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim

    def preprocess(self):
        self.adjust_reshape_and_expand()
        return

    def adjust_reshape_and_expand(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Reshape":
                # Clean up unneccessary reshape nodes.
                # Find reshape nodes with no actually data in "shape" attribute and remove.
                reshape_shape = self.get_constant_value(node.input[1])
                if reshape_shape is not None and reshape_shape.size == 0:
                    nodes_to_remove.extend([node])
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    continue

                # Find path "Slice" -> "Reshape" -> "Expand" -> "Expand" -> current "Reshape", simplify the graph by
                # changing current reshape's input to output of slice.
                reshape_path = self.match_parent_path(
                    node,
                    ["Expand", "Expand", "Reshape", "Slice"],
                    [0, 0, 0, 0],
                    self.output_name_to_node(),
                )
                if reshape_path is not None:
                    expand_node = reshape_path[-3]
                    expand_shape_value = self.get_constant_value(expand_node.input[1])

                    reshape_before_expand = reshape_path[-2]
                    shape_value = self.get_constant_value(
                        reshape_before_expand.input[1]
                    )

                    slice_node = reshape_path[-1]
                    if (
                        expand_shape_value is not None
                        and shape_value is not None
                        and len(expand_shape_value) == 2
                        and len(shape_value) == 1
                        and expand_shape_value[1] == shape_value[0]
                    ):
                        node.input[0] = slice_node.output[0]

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info(f"Removed Reshape and Expand count: {len(nodes_to_remove)}")

    def clean_graph(self):
        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []
        for node in self.nodes():
            # Before:
            #  input_ids --> Shape --> Gather(indices=0) --> Unsqueeze ------+
            #          |                                                     |
            #          |                                                     v
            #          +----> Shape --> Gather(indices=1) --> Unsqueeze--->  Concat --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # After:
            #  input_ids --> Shape                                                  --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # TODO: merge ConstantOfShape -->Cast to ConstantOfShape (need update the data type of value)
            op_input_id = {"EmbedLayerNormalization": 1, "ReduceSum": 0, "Attention": 3}
            if node.op_type in op_input_id:
                i = op_input_id[node.op_type]
                parent_nodes = self.match_parent_path(
                    node,
                    [
                        "Cast",
                        "ConstantOfShape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [i, 0, 0, 0, 0, 0],
                    output_name_to_node,
                )
                if parent_nodes is not None:
                    (
                        cast,
                        constantOfShape,
                        concat,
                        unsqueeze,
                        gather,
                        shape,
                    ) = parent_nodes
                    if shape.input[0] == self.graph().input[0].name:
                        constantOfShape.input[0] = shape.output[0]
                        output_name_to_node = self.output_name_to_node()

            if node.op_type == "Attention":
                # Before:
                #   input_ids --> Shape -->ConstantOfShape -->Cast --> ReduceSum --> Attention
                # After:
                #   remove this path, and remove the optional mask_index input of Attention node.
                parent_nodes = self.match_parent_path(
                    node,
                    ["ReduceSum", "Cast", "ConstantOfShape", "Shape"],
                    [3, 0, 0, 0],
                    output_name_to_node,
                )
                if parent_nodes is not None:
                    if parent_nodes[-1].input[0] == self.graph().input[0].name:
                        attention_node = helper.make_node(
                            "Attention",
                            inputs=node.input[0 : len(node.input) - 1],
                            outputs=node.output,
                            name=node.name + "_remove_mask",
                        )
                        attention_node.domain = "com.microsoft"
                        attention_node.attribute.extend(
                            [helper.make_attribute("num_heads", self.num_heads)]
                        )
                        self.add_node(
                            attention_node, self.get_graph_by_node(attention_node).name
                        )
                        nodes_to_remove.append(node)
        self.remove_nodes(nodes_to_remove)

    def postprocess(self):
        self.clean_graph()
        self.prune_graph()

    def optimize(
        self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False
    ):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        if options.enable_swint_opt:
            self.fuse_custom_fc()
            self.fuse_swinT_serial_bias_add()

        if options.enable_format_roformer:
            self.fuse_format_roformer()

        if options.enable_gpt2_classify or options.enable_vit:
            self.fuse_custom_fc_gpt2_classify()

        if options.enable_vit:
            self.fuse_custom_fc()

        if (options is None) or options.enable_attention:
            if options is not None:
                self.attention_mask.set_mask_format(options.attention_mask_format)
            self.fuse_attention()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        self.fuse_custom_fc()
        
        if options.enable_omdet:
            self.fuse_omdet_attention()
            self.fuse_omdet_inverse_sigmoid()
            # self.fuse_dino_inverse_sigmoid()
            self.fuse_l2_normalization()

        self.fuse_custom_xsoftmax()

        self.fuse_disentangled_attention()

        # Perform the MatMul fusion after the Attention fusion as we do not
        # want to fuse the MatMuls inside the Attention subgraphs
        if (options is None) or options.enable_qordered_matmul:
            self.fuse_qordered_mamtul()

        self.fuse_shape()

        if (options is None) or options.enable_embed_layer_norm:
            self.fuse_embed_layer()

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()

        self.postprocess()

        # Bias fusion is done after postprocess to avoid extra Reshape between bias and Gelu/FastGelu/SkipLayerNormalization
        if (options is None) or options.enable_bias_gelu:
            # Fuse Gelu and Add Bias before it.
            self.fuse_bias_gelu(is_fastgelu=True)
            self.fuse_bias_gelu(is_fastgelu=False)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        if options is not None and options.enable_gelu_approximation:
            self.gelu_approximation()

        self.fuse_custom_fc_activation()
        
        if options.enable_vit:
            self.fuse_custom_fc_torchvision_vit()

        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        if add_dynamic_axes:
            self.use_dynamic_axes()

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "EmbedLayerNormalization",
            "Attention",
            "QOrderedAttention",
            "Gelu",
            "QOrderedGelu",
            "FastGelu",
            "BiasGelu",
            "LayerNormalization",
            "QOrderedLayerNormalization",
            "SkipLayerNormalization",
            "QOrderedMatMul",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)
        logger.info(f"Optimized operators:{op_count}")
        return op_count

    def is_fully_optimized(self):
        """
        Returns True when the model is fully optimized.
        """
        op_count = self.get_fused_operator_statistics()
        embed = op_count["EmbedLayerNormalization"]
        attention = op_count["Attention"] + op_count["QOrderedAttention"]
        gelu = op_count["Gelu"] + op_count["BiasGelu"] + op_count["FastGelu"]
        layer_norm = op_count["LayerNormalization"] + op_count["SkipLayerNormalization"]
        is_perfect = (
            (embed > 0)
            and (attention > 0)
            and (attention == gelu)
            and (layer_norm >= 2 * attention)
        )

        if layer_norm == 0:
            logger.debug("Layer Normalization not fused")

        if gelu == 0:
            logger.debug("Gelu/FastGelu not fused")

        if embed == 0:
            logger.debug("Embed Layer not fused")

        if attention == 0:
            logger.warning("Attention not fused")

        return is_perfect
