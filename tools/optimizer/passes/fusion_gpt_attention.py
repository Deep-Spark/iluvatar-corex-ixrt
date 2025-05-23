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
from onnx import TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import FusionUtils
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGptAttentionPastBase(Fusion):
    """Base class for GPT Attention Fusion with past state"""

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", "LayerNormalization", "with past")
        self.num_heads = num_heads
        self.utils = FusionUtils(model)
        self.casted_attention_mask = (
            {}
        )  # map from name of attention mask to the name that casted to int32

    def match_past_pattern_1(self, concat_k, concat_v, output_name_to_node):
        # Pattern 1:
        #                      {past}
        #                    /        \
        #                   /          \
        #    Gather(axes=0, indices=0)  Gather(indices=1)
        #      |                          |
        #    Transpose (perm=0,1,3,2)     |
        #      |                          |
        #  Concat_k                     Concat_v
        #      |                        /
        #  Transpose (perm=0,1,3,2)    /
        #      |                      /
        #  Unsqueeze        Unsqueeze
        #        \        /
        #         \      /
        #           Concat
        #             |
        #         {present}
        gather = self.model.get_parent(concat_v, 0, output_name_to_node)
        if gather.op_type != "Gather":
            logger.debug("match_past_pattern_1: expect Gather for past")
            return None

        if not self.model.find_constant_input(gather, 1) == 1:
            logger.debug("match_past_pattern_1: expect indices=1 for Gather of past")
            return None
        past = gather.input[0]

        parent = self.model.get_parent(concat_k, 0, output_name_to_node)
        if parent.op_type == "Gather":
            gather_past_k = parent
        else:
            past_k_nodes = self.model.match_parent_path(
                concat_k, ["Transpose", "Gather"], [0, 0]
            )
            if past_k_nodes is None:
                logger.debug("match_past_pattern_1: failed match Transpose and Gather")
                return None
            gather_past_k = past_k_nodes[-1]

        if not self.model.find_constant_input(gather_past_k, 0) == 1:
            logger.debug("match_past_pattern_1: expect indices=0 for Gather k of past")
            return None
        past_k = gather_past_k.input[0]
        if past != past_k:
            logger.debug("match_past_pattern_1: expect past to be same")
            return None

        return past

    def match_past_pattern_2(self, concat_k, concat_v, output_name_to_node):
        # Pattern 2:
        #      Split (QKV)
        #      / |   |
        #     /  |   +----------------------+
        #        |                          |
        #        |         {past}           |
        #        |           |              |
        #      Reshape     Split         Reshape
        #        |         /    \           |
        # Transpose_k  Squeeze  Squeeze  Transpose_v
        #        |      |        \        /
        #        +------|---+     \      /
        #               |   |      \    /
        #              Concat_k   Concat_v
        #               |            |
        #          Unsqueeze    Unsqueeze
        #                \       /
        #                 Concat
        #                   |
        #               {present}
        #
        squeeze = self.model.get_parent(concat_v, 0, output_name_to_node)
        if squeeze.op_type != "Squeeze":
            logger.debug("match_past_pattern_2: expect Squeeze as parent of concat_v")
            return None

        split = self.model.get_parent(squeeze, 0, output_name_to_node)
        if split.op_type != "Split":
            logger.debug("match_past_pattern_2: expect Split for past path")
            return None

        opset_version = self.model.get_opset_version()
        if opset_version < 13:
            if not FusionUtils.check_node_attribute(squeeze, "axes", [0]):
                logger.debug(
                    "match_past_pattern_2: axes != [0] for Squeeze in past path"
                )
                return None

            if not FusionUtils.check_node_attribute(split, "split", [1, 1]):
                logger.debug(
                    "match_past_pattern_2: split != [1, 1] for Split in past path"
                )
                return None
        else:
            if not self.utils.check_node_input_value(squeeze, 1, [0]):
                logger.debug(
                    "match_past_pattern_2: axes != [0] for Squeeze in past path"
                )
                return None

            if not self.utils.check_node_input_value(split, 1, [1, 1]):
                logger.debug(
                    "match_past_pattern_2: split != [1, 1] for Split in past path"
                )
                return None

        if not FusionUtils.check_node_attribute(split, "axis", 0, default_value=0):
            logger.debug(
                "match_past_pattern_2: attribute axis of Split are not expected in past path"
            )
            return None
        past = split.input[0]

        past_k_nodes = self.model.match_parent_path(
            concat_k, ["Squeeze", "Split"], [0, 0]
        )
        if past_k_nodes is None:
            logger.debug("match_past_pattern_2: failed to match past_k_nodes path")
            return None
        past_k = past_k_nodes[-1].input[0]

        if past != past_k:
            logger.info("match_past_pattern_2: expect past to be same")
            return None

        return past

    def match_present(self, concat_v, input_name_to_nodes):
        unsqueeze_present_v = self.model.find_first_child_by_type(
            concat_v, "Unsqueeze", input_name_to_nodes, recursive=False
        )
        if not unsqueeze_present_v:
            logger.info("expect unsqueeze for present")
            return None
        concat_present = self.model.find_first_child_by_type(
            unsqueeze_present_v, "Concat", input_name_to_nodes, recursive=False
        )
        if not concat_present:
            logger.info("expect concat for present")
            return None

        present = concat_present.output[0]
        return present

    def cast_attention_mask(self, input_name):
        if input_name in self.casted_attention_mask:
            attention_mask_input_name = self.casted_attention_mask[input_name]
        elif self.model.find_graph_input(input_name):
            casted, attention_mask_input_name = self.utils.cast_graph_input_to_int32(
                input_name
            )
            self.casted_attention_mask[input_name] = attention_mask_input_name
        else:
            attention_mask_input_name, cast_node = self.utils.cast_input_to_int32(
                input_name
            )
            self.casted_attention_mask[input_name] = attention_mask_input_name
        return attention_mask_input_name


class FusionGptAttention(FusionGptAttentionPastBase):
    """
    Fuse GPT-2 Attention with past state subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, num_heads)

    def create_attention_node(
        self,
        fc_weight,
        fc_bias,
        gemm_qkv,
        past,
        present,
        input,
        output,
        mask,
        is_unidirectional,
    ):
        attention_node_name = self.model.create_node_name("GptAttention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, fc_weight, fc_bias, mask, past],
            outputs=[attention_node_name + "_output", present],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1 if is_unidirectional else 0),
            ]
        )

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
            outputs=[attention_node_name + "_matmul_output"],
            name=attention_node_name + "_matmul",
        )

        add_node = helper.make_node(
            "Add",
            inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
            outputs=[output],
            name=attention_node_name + "_add",
        )
        self.nodes_to_add.extend([attention_node, matmul_node, add_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        past = None
        present = None
        return_indice = []
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
            [0, None, 0, 0, 0, 0, 0],
            output_name_to_node=output_name_to_node,
            return_indice=return_indice,
        )  # yapf: disable
        if qkv_nodes is None:
            return
        (
            add_qkv,
            reshape_qkv,
            gemm_qkv,
            reshape_1,
            reshape_2,
            transpose_qkv,
            matmul_qkv,
        ) = qkv_nodes

        another_input = add_qkv.input[1 - return_indice[0]]

        v_nodes = self.model.match_parent_path(
            matmul_qkv, ["Concat", "Transpose", "Reshape", "Split"], [1, 1, 0, 0]
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (concat_v, transpose_v, reshape_v, split_fc) = v_nodes

        fc_nodes = self.model.match_parent_path(
            split_fc,
            ["Reshape", "Gemm", "Reshape", "LayerNormalization"],
            [0, 0, 0, 0],
            output_name_to_node,
        )
        if fc_nodes is None:
            fc_nodes = self.model.match_parent_path(
                split_fc,
                ["Add", "MatMul", "LayerNormalization"],
                [0, None, 0],
                output_name_to_node,
            )
            if fc_nodes is None:
                logger.debug("fuse_attention: failed to match fc path")
                return
            fc_weight = fc_nodes[1].input[1]
            i, _ = self.model.get_constant_input(fc_nodes[0])
            fc_bias = fc_nodes[0].input[i]
        else:
            fc_weight = fc_nodes[1].input[1]
            fc_bias = fc_nodes[1].input[2]

        layernorm_before_attention = fc_nodes[-1]

        if not another_input in layernorm_before_attention.input:
            logger.debug("Add and LayerNormalization shall have one same input")
            return

        is_unidirectional = True
        slice_mask = None
        input_mask_nodes = None
        concat_k_to_match = None
        qk_nodes = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "Sub", "Mul", "Div", "MatMul"], [0, 0, 0, 0, 0]
        )
        if qk_nodes is not None:
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                [
                    "Mul",
                    "Sub",
                    "Slice",
                    "Slice",
                    "Unsqueeze",
                    "Sub",
                    "Squeeze",
                    "Slice",
                    "Shape",
                    "Div",
                ],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            )  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match unidirectional mask path")
                return
            div_mask = mask_nodes[-1]
            slice_mask = mask_nodes[3]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return
        else:
            # New pattern for gpt2 from PyTorch 1.5.0 and Transformers 2.9.0.
            i, qk_nodes, _ = self.model.match_parent_paths(
                matmul_qkv,
                [
                    (["Softmax", "Where", "Div", "MatMul"], [0, 0, 1, 0]),
                    (["Softmax", "Add", "Where", "Div", "MatMul"], [0, 0, None, 1, 0]),
                ],
                output_name_to_node,
            )
            if qk_nodes is None:
                logger.debug("fuse_attention: failed to match qk nodes")
                return

            where_qk = qk_nodes[-3]
            div_qk = qk_nodes[-2]
            matmul_qk = qk_nodes[-1]

            if i == 1:
                add_qk = qk_nodes[1]
                _, input_mask_nodes, _ = self.model.match_parent_paths(
                    add_qk,
                    [
                        (
                            ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze", "Reshape"],
                            [None, 0, 1, 0, 0, 0],
                        ),
                        (
                            ["Mul", "Sub", "Unsqueeze", "Unsqueeze", "Reshape"],
                            [None, 0, 1, 0, 0],
                        ),
                        (
                            ["Mul", "Sub", "Unsqueeze", "Unsqueeze"],
                            [None, 0, 1, 0],
                        ),  # useless cast and reshape are removed.
                    ],
                    output_name_to_node,
                )  # yapf: disable
                if input_mask_nodes is None:
                    logger.debug(
                        "fuse_attention: failed to match input attention mask path"
                    )
                    return

            mask_nodes = self.model.match_parent_path(
                where_qk,
                [
                    "Cast",
                    "Slice",
                    "Slice",
                    "Unsqueeze",
                    "Sub",
                    "Squeeze",
                    "Slice",
                    "Shape",
                ],
                [0, 0, 0, 1, 0, 0, 0, 0],
                output_name_to_node,
            )  # yapf: disable
            if mask_nodes is None:
                # TODO: match mask path for GPT2LMHeadModel_BeamSearchStep.
                logger.debug("fuse_attention: failed to match mask path")
                return

            slice_mask = mask_nodes[2]

            div_or_concat = self.model.get_parent(
                mask_nodes[-1], 0, output_name_to_node
            )
            if div_or_concat.op_type == "Div":
                div_mask = div_or_concat
                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return
            elif div_or_concat.op_type == "Concat":
                concat_k_to_match = div_or_concat
            else:
                logger.debug("fuse_attention: failed to match mask path")

        # Validate that the mask data is either lower triangular (unidirectional) or all ones
        mask_data = numpy_helper.to_array(
            self.model.get_initializer(slice_mask.input[0])
        )
        if not (
            len(mask_data.shape) == 4
            and mask_data.shape[:2] == (1, 1)
            and mask_data.shape[2] == mask_data.shape[3]
        ):
            logger.debug("fuse_attention: skip since mask shape is not 1x1xWxW")
            return
        if np.allclose(mask_data, np.ones_like(mask_data)):
            is_unidirectional = False
        elif not np.allclose(mask_data, np.tril(np.ones_like(mask_data))):
            logger.debug(
                "fuse_attention: skip since mask is neither lower triangular nor ones"
            )
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Split"], [0, 0, 0]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_fc != split_q:
            logger.debug("fuse_attention: skip since split_fc != split_q")
            return

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Concat", "Transpose", "Reshape", "Split"], [1, 1, 0, 0]
        )
        if k_nodes is None:
            # This pattern is from pytorch 1.7.1 and transformers 4.6.1
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Concat", "Transpose", "Reshape", "Split"],
                [1, 0, 1, 0, 0],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
            else:
                (_, concat_k, transpose_k, reshape_k, split_k) = k_nodes
        else:
            (concat_k, transpose_k, reshape_k, split_k) = k_nodes
        if split_fc != split_k:
            logger.debug("fuse_attention: skip since split_fc != split_k")
            return

        if concat_k_to_match and concat_k != concat_k_to_match:
            logger.debug("fuse_attention: skip since concat_k != concat_k_to_match")
            return

        attention_mask_input_name = ""
        if input_mask_nodes is not None:
            input_name = input_mask_nodes[-1].input[0]
            attention_mask_input_name = self.cast_attention_mask(input_name)

        # Match past and present paths
        past = self.match_past_pattern_1(
            concat_k, concat_v, output_name_to_node
        ) or self.match_past_pattern_2(concat_k, concat_v, output_name_to_node)
        if past is None:
            logger.info("fuse_attention: failed to match past path")
            return
        if not self.model.find_graph_input(past):
            logger.debug("past is not graph input.")
            # For GPT2LMHeadModel_BeamSearchStep, there is an extra Gather node to select beam index so it is not graph input.

        present = self.match_present(concat_v, input_name_to_nodes)
        if present is None:
            logger.info("fuse_attention: failed to match present path")
            return
        if not self.model.find_graph_output(present):
            logger.info("expect present to be graph output")
            return

        self.create_attention_node(
            fc_weight,
            fc_bias,
            gemm_qkv,
            past,
            present,
            layernorm_before_attention.output[0],
            reshape_qkv.output[0],
            attention_mask_input_name,
            is_unidirectional,
        )

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True
