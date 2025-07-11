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
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import List, Tuple, Union

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_attention import AttentionMask
from .fusion_base import Fusion
from .fusion_options import AttentionMaskFormat
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel
from .shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)


def get_tensor_attr(attrs, attr_name):
    result = None
    for i in attrs:
        if i.name == attr_name:
            return numpy_helper.to_array(i.t)
    return result


class FusionAlbertAttention(Fusion):
    """
    Fuse Albert subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(
            model,
            "CustomQKVToContextPluginDynamic_IxRT",
            ["CustomSkipLayerNormPluginDynamic_IxRT", "LayerNormalization"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape_value = self.model.get_constant_value(reshape_q.input[1])
        if q_shape_value is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(
                f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size]."
            )
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(
                    f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value."
                )
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = (
                    False  # Do not show the warning more than once
                )

        return num_heads, hidden_size

    def get_add_qk_str(self, add_qk: NodeProto):
        shape_infer = self.model.infer_runtime_shape(update=True)
        if shape_infer is None:
            return

        input_0_shape = shape_infer.get_edge_shape(add_qk.input[0])
        input_1_shape = shape_infer.get_edge_shape(add_qk.input[1])

        if input_0_shape is None or input_1_shape is None:
            logger.debug(f"one of the inputs of {add_qk} is None")
            return None

        if input_0_shape != input_1_shape:
            logger.debug(f"the shape of two inputs of {add_qk} is not same")
            return None

        return add_qk.input[1]

    def create_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for  K
            v_matmul (NodeProto): MatMul node in fully connection for  V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(
                f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}"
            )
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        q_bias = self.model.get_initializer(
            q_add.input[1]
        ) or self.model.get_initializer(q_add.input[0])
        k_bias = self.model.get_initializer(
            k_add.input[1]
        ) or self.model.get_initializer(k_add.input[0])
        v_bias = self.model.get_initializer(
            v_add.input[1]
        ) or self.model.get_initializer(v_add.input[0])

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None
        if not (k_weight and v_weight and q_bias and k_bias):
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        is_qkv_diff_dims = False

        # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        qkv_weight_dim = 0
        qkv_weight = np.concatenate((qw, kw, vw), axis=1)
        qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size

        qb = NumpyHelper.to_array(q_bias)
        kb = NumpyHelper.to_array(k_bias)
        vb = NumpyHelper.to_array(v_bias)

        q_bias_shape = np.prod(qb.shape)
        k_bias_shape = np.prod(kb.shape)
        v_bias_shape = np.prod(vb.shape)

        assert q_bias_shape == k_bias_shape == qw_out_size
        assert v_bias_shape == vw_out_size

        qkv_bias_dim = 0
        if is_qkv_diff_dims:
            qkv_bias = np.concatenate((qb, kb, vb), axis=0)
            qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape
        else:
            qkv_bias = np.stack((qb, kb, vb), axis=0)
            qkv_bias_dim = 3 * q_bias_shape

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[qkv_weight_dim, qw_in_size],
            vals=qkv_weight.transpose(1, 0).flatten().tolist(),
        )

        # Sometimes weights and bias are stored in fp16
        if q_weight.data_type == 10:
            weight.CopyFrom(
                numpy_helper.from_array(
                    NumpyHelper.to_array(weight).astype(np.float16), weight.name
                )
            )
        self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkv_bias.flatten().tolist(),
        )
        if q_bias.data_type == 10:
            bias.CopyFrom(
                numpy_helper.from_array(
                    NumpyHelper.to_array(bias).astype(np.float16), bias.name
                )
            )
        self.model.add_initializer(bias, self.this_graph_name)

        fc_output_tensor = helper.make_tensor_value_info(
            attention_node_name + "_input", TensorProto.FLOAT, [None, None, None]
        )
        fc_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[input],
            outputs=[fc_output_tensor.name],
            name=self.model.create_node_name("AttentionFC", "MatMul_AddBias_"),
        )
        fc_node.domain = "com.iluvatar"
        b = NumpyHelper.to_array(bias)
        fc_node.attribute.extend([helper.make_attribute("out_dims", b.shape[0])])
        fc_node.attribute.extend([helper.make_attribute("type_id", 1)])
        fc_node.attribute.extend([helper.make_attribute("W", weight)])
        fc_node.attribute.extend([helper.make_attribute("B", bias)])
        fc_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fc_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fc_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fc_node.name] = self.this_graph_name
        self.nodes_to_add.append(fc_node)

        attention_inputs = [fc_node.output[0]]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if add_qk_str is not None:
            attention_inputs.append("")
            attention_inputs.append(add_qk_str)

        attention_node = helper.make_node(
            "CustomQKVToContextPluginDynamic_IxRT",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 1)])
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend(
            [helper.make_attribute("hidden_size", hidden_size)]
        )
        attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 1)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [
                    helper.make_attribute(
                        "qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size]
                    )
                ]
            )

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, 0, 0, 0],
        )
        if qkv_nodes is None:
            qkv_nodes = self.model.match_parent_path(
                start_node,
                ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
                [1, None, 0, 0, 0],
            )
        einsum_node = None
        if qkv_nodes is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            # Match Albert
            qkv_nodes = self.model.match_parent_path(
                start_node, ["Add", "Einsum", "Transpose", "MatMul"], [1, None, 0, 0]
            )
            if qkv_nodes is not None:
                (_, einsum_node, transpose_qkv, matmul_qkv) = qkv_nodes
            else:
                return

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        """
        Match flaubert                     Mask
                                            |
        Mul --> LayerNormalization -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        mul_before_layernorm = self.model.match_parent(start_node, "Mul", 0)
        if mul_before_layernorm is not None:
            mul_children = input_name_to_nodes[mul_before_layernorm.output[0]]
            if mul_children is not None and len(mul_children) == 2:
                layernorm_node = mul_children[1]
                if layernorm_node.op_type == "LayerNormalization":
                    root_input = layernorm_node.output[0]
                else:
                    return
            elif mul_children is not None and len(mul_children) == 5:
                root_input = mul_before_layernorm.output[0]
            else:
                return
        elif normalize_node.op_type == "LayerNormalization":
            children = input_name_to_nodes[root_input]
            for child in children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count("MatMul") != 3:
            return

        v_nodes = self.model.match_parent_path(
            matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None]
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        is_distill = False
        is_distill_add = False
        is_mul_split = False
        qk_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul"], [0, 0, None, 0]),
            "path2": (["Softmax", "Add", "Mul", "MatMul"], [0, 0, None, 0]),
            "path3": (["Softmax", "Where", "MatMul", "Div"], [0, 0, 2, 0]),
            "path4": (["Softmax", "Add", "Where", "MatMul"], [0, 0, 0, 2]),
            "path5": (["Softmax", "Add", "MatMul"], [0, 0, None])
        }

        qk_nodes = None
        for k, v in qk_paths.items():
            qk_nodes = self.model.match_parent_path(matmul_qkv, v[0], v[1])
            if qk_nodes is None:
                continue
            if k == "path3":
                is_distill = True
            if k == "path4":
                is_distill_add = True
            if k == "path5":
                is_mul_split = True
            break

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return
        add_qk = None
        matmul_qk = None
        where_qk = None
        if is_distill:
            (_, where_qk, matmul_qk, _) = qk_nodes
        elif is_distill_add:
            (_, add_qk, where_qk, matmul_qk) = qk_nodes
        elif is_mul_split:
            (_, add_qk, matmul_qk) = qk_nodes
        else:
            (_, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, None]
        )
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Div", "Transpose", "Reshape", "Add", "MatMul"],
                [0, 0, 0, 0, None],
            )
            if q_nodes is None and is_mul_split:
                q_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Mul", "Transpose", "Reshape", "Add", "MatMul"],
                    [0, 0, 0, 0, None],
                )
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
        reshape_q = q_nodes[-3]
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None]
        )
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Transpose", "Reshape", "Add", "MatMul"],
                [1, 0, 0, 0, None],
            )
            if k_nodes is None and is_mul_split:
                k_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Mul", "Transpose", "Reshape", "Add", "MatMul"],
                    [1, 0, 0, 0, None],
                )
            
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        mask_nodes = None
        add_qk_str = None
        if is_distill:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Expand", "Reshape", "Equal"], [0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                    (["Cast", "Expand", "Reshape", "Equal"], [0, 0, 0, 0]),
                ],
                output_name_to_node,
            )
        elif is_distill_add:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Cast", "Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                ],
                output_name_to_node,
            )
            if add_qk is not None:
                add_qk_str = self.get_add_qk_str(add_qk)
                if add_qk_str is None:
                    logger.debug(
                        f"fuse_attention: failed to verify shape inference of {add_qk}"
                    )
                    return
        elif is_mul_split:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (["Where", "Cast", "Sub", "Cast", "Expand", "Unsqueeze"], [None, 0, 0, 1, 0, 0])
                ],
                output_name_to_node,
            )
        else:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (
                        ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                        [None, 0, 1, 0, 0],
                    ),
                    (["Mul", "Sub", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0]),
                    (["Mul", "Sub", "Cast", "Unsqueeze"], [None, 0, 1, 0]),
                ],
                output_name_to_node,
            )
        if mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if (
            matmul_v.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == root_input
        ):
            # mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])
            if mask_nodes[0].op_type == "Mul":
                mask_val = self.model.get_initializer(mask_nodes[0].input[1])
                if mask_val is not None:
                    mask_val_arr = NumpyHelper.to_array(mask_val)
                    mask_val_arr = np.where(mask_val_arr <= -100, -100, 0.0).astype(
                        np.float32
                    )
                    mask_val.CopyFrom(
                        numpy_helper.from_array(mask_val_arr, mask_val.name)
                    )
            mask_index = mask_nodes[0].output[0]

            attention_last_node = reshape_qkv if einsum_node is None else transpose_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_attention_node(
                mask_index,
                matmul_q,
                matmul_k,
                matmul_v,
                add_q,
                add_k,
                add_v,
                q_num_heads,
                q_hidden_size,
                root_input,
                attention_last_node.output[0],
                add_qk_str,
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            if einsum_node is not None:
                unique_index = einsum_node.input[0]
                new_edge = "edge_modified_" + unique_index
                shape_tensor = helper.make_tensor(
                    name="shape_modified_tensor" + unique_index,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=np.int64(
                        [0, 0, q_num_heads, int(q_hidden_size / q_num_heads)]
                    ).tobytes(),
                    raw=True,
                )
                self.model.add_initializer(shape_tensor, self.this_graph_name)
                self.model.add_node(
                    helper.make_node(
                        "Reshape",
                        [attention_last_node.output[0], shape_tensor.name],
                        [new_edge],
                        "reshape_modified_" + unique_index,
                    ),
                    self.this_graph_name,
                )
                einsum_node.input[0] = new_edge

            self.nodes_to_remove.extend(
                [attention_last_node, transpose_qkv, matmul_qkv]
            )
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True
