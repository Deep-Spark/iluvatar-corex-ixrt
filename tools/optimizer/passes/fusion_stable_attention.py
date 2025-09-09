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


class FusionStableAttention(Fusion):
    """
    Fuse Albert subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int
    ):
        super().__init__(
            model,
            "CustomQKVToContextPluginDynamic_IxRT",
            ["LayerNormalization"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads

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
        add_before_layernorm = None
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, None, 0, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (_, _, reshape_qkv, transpose_qkv, _, matmul_qkv) = qkv_nodes
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
        Match Stable fusion 
                                            |
        Add --> LayerNormalization -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        add_children = input_name_to_nodes[root_input]
        if add_children is not None and len(add_children) == 2:
            update_input = False
            for child in add_children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]
                    update_input = True
            if not update_input:
                return
        else:
            return

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count("MatMul") != 3:
            return

        v_nodes = self.model.match_parent_path(
            matmul_qkv, ["Reshape", "Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, None]
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, _, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv,["Softmax", "Reshape", "Add", "Reshape", "MatMul"], [0, 0, 0, 0, None])
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return
        add_qk = None
        matmul_qk = None
        where_qk = None
        (_, _, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Reshape", "Transpose", "Reshape", "Mul", "Add", "MatMul"], [0, 0, 0, 0, 0, None]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        reshape_q = q_nodes[-4]
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, 0, None]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        add_qk_str = None
        mask_node = self.model.match_parent(add_qk, "Expand", 1)
        if mask_node is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if (
            matmul_v.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == root_input
        ):
            mask_index = mask_node.output[0]

            attention_last_node = reshape_qkv 
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
