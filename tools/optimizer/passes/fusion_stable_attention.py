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
import random

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
            ["LayerNormalization", "Add"],
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


    def make_value_info_from_tensor(self, tensor):
        return helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)

    def try_to_get_num_heads_and_hidden_size_for_self_attention(self, matmu_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a matmul node.

        Args:
            matmu_q (NodeProto): matmul node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        if not matmu_q:
            logger.debug(f"matmu_q is empty.")

        # in unet self attention, hidden_size / num_heads = 64
        div_num = 64

        all_value = (
            list(self.model.model.graph.value_info) +
            list(self.model.model.graph.output) +
            list(self.model.model.graph.input) +
            [self.make_value_info_from_tensor(t) for t in self.model.model.graph.initializer]
        )
        val_info  = next(v for v in all_value if v.name == matmu_q.input[1])
        shape = []
        for d in val_info.type.tensor_type.shape.dim:
            if d.HasField('dim_value'):
                shape.append(d.dim_value)
            elif d.HasField('dim_param'):
                shape.append(-1)   # dynamic
        else:
            shape.append(-2)

        if len(shape) < 2 or shape[0] < 0 or (shape[0] % div_num != 0):
            logger.debug(f"matmul_q have wrong input1 of ", val_info)
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        return shape[0] // div_num, shape[0]


    def create_fuse_matmul_node(self,
        input: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto
    ):
        q_weight = onnx.numpy_helper.to_array(self.model.get_initializer(q_matmul.input[1]))
        k_weight = onnx.numpy_helper.to_array(self.model.get_initializer(k_matmul.input[1]))
        v_weight = onnx.numpy_helper.to_array(self.model.get_initializer(v_matmul.input[1]))
    
        # 在水平方向拼接权重
        concatenated_weights = np.concatenate([q_weight, k_weight, v_weight], axis=1)
    
        random_idx_str = str(random.randint(1, 100000) * random.randint(1, 100000))
        # 创建新的权重初始化器
        new_weight_name = "fused_weights_of_input_" + input + '_' + random_idx_str
        new_weight_tensor = numpy_helper.from_array(
            concatenated_weights, name=new_weight_name
        )

        self.model.add_initializer(new_weight_tensor, self.this_graph_name)
    
        # 创建新的MatMul节点
        new_matmul_name = self.model.create_node_name("qkv_fused_matmul")
        output_tensor_name = input + "_cord_fuse_matmul_output_" + random_idx_str
        new_matmul_node = helper.make_node(
            "MatMul",
            inputs=[input, new_weight_name],
            outputs=[output_tensor_name],
            name=new_matmul_name
        )
        return new_matmul_node
        
    def create_attention_node(
        self,
        attention_node_name: str,
        mask_index: str,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str, 
        is_qkv_diff_dims: bool, 
        qkv_weight_size) -> Union[NodeProto, None]:

        attention_inputs = [input]
        if mask_index is not None:
            attention_inputs.append(mask_index)

        if add_qk_str is not None:
            if mask_index is None:
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
        if mask_index is not None:
            attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        else:
            attention_node.attribute.extend([helper.make_attribute("has_mask", 0)]) 
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        if mask_index is not None:
            attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 1)])
        else:
            attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 0)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [
                    helper.make_attribute(
                        "qkv_hidden_sizes", qkv_weight_size 
                    )
                ]
            )

        return attention_node
    
    def create_matmul_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str, 
    ) -> Union[NodeProto, None]:

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]
        assert qw_in_size == kw_in_size == vw_in_size


        fused_matmul_node = self.create_fuse_matmul_node(input, q_matmul, k_matmul, v_matmul)
        self.node_name_to_graph_name[fused_matmul_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_matmul_node)

        attention_node_name = self.model.create_node_name("Attention")

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        
        return self.create_attention_node(attention_node_name, mask_index, num_heads, hidden_size, fused_matmul_node.output[0], output, add_qk_str, False, None)
    


    def create_qkv_attention_node(
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
        if qw.shape != vw.shape:
            is_qkv_diff_dims = True

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

        return self.create_attention_node(attention_node_name, mask_index, num_heads, hidden_size, fc_node.output[0], output, add_qk_str, is_qkv_diff_dims, [qw_out_size, kw_out_size, vw_out_size])
    
    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type == "LayerNormalization":
            self.fuse1(node, input_name_to_nodes, output_name_to_node)
        elif node.op_type == "Add":
            self.fuse2(node, input_name_to_nodes, output_name_to_node)

    def fuse1(self, normalize_node, input_name_to_nodes, output_name_to_node):
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
        match_id, qkv_nodes, _ = self.model.match_parent_paths(
            start_node,
            [
                (["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
                 [1, None, 0, 0, 0, 0]),
                (["Add", "MatMul", "Reshape", "Transpose",  "MatMul"],
                 [0, None, 0, 0, 0]),
            ],
            output_name_to_node
        )
        if qkv_nodes is not None:
            if match_id == 0:
                (_, _, reshape_qkv, transpose_qkv, _, matmul_qkv) = qkv_nodes
            else:
                (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
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

        match_id, v_nodes, _ = self.model.match_parent_paths(
            matmul_qkv, 
            [
                (["Reshape", "Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, None]),
                (["Transpose", "Reshape", "MatMul"], [1, 0, None]),
            ],
            output_name_to_node
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        if match_id == 0:
            (_, _, _, add_v, matmul_v) = v_nodes
        else:
            (_, _, matmul_v) = v_nodes
            add_v = None

        match_id, qk_nodes, _ = self.model.match_parent_paths(
            matmul_qkv,
            [
                (["Softmax", "Reshape", "Add", "Reshape", "MatMul"], [0, 0, 0, 0, None]),
                (["Softmax", "MatMul"], [0, None]),
            ],
            output_name_to_node)
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        add_qk = None
        matmul_qk = None
        if match_id == 0:
            (_, _, add_qk, _, matmul_qk) = qk_nodes
        else:
            (_,  matmul_qk) = qk_nodes

        match_id, q_nodes, _ = self.model.match_parent_paths(
            matmul_qk,
            [ 
                (["Reshape", "Transpose", "Reshape", "Mul", "Add", "MatMul"], [0, 0, 0, 0, 0, None]),
                (["Mul", "Transpose", "Reshape", "MatMul"], [0, 0, 0, None]),
            ],
            output_name_to_node
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return

        reshape_q = None
        add_q = None
        matmul_q = None
        if match_id == 0:
            (_, _, reshape_q, _, add_q, matmul_q) = q_nodes
        else:   
            (_, _, reshape_q, matmul_q) = q_nodes

        match_id, k_nodes, _ = self.model.match_parent_paths(
            matmul_qk, 
            [
                (["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, 0, None]),
                (["Mul", "Transpose", "Reshape", "MatMul"], [1, 0, 0, None])
            ],
            output_name_to_node
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        
        add_k = None
        matmul_k = None

        if match_id == 0:
            (_, _, _, _, add_k, matmul_k) = k_nodes
        else:   
            (_, _, _, matmul_k) = k_nodes

        add_qk_str = None
        mask_index = None
        if add_qk is not None:
            mask_node = self.model.match_parent(add_qk, "Expand", 1)
            if mask_node is None:
                logger.debug("fuse_attention: failed to match mask path")
                return
            else:
                mask_index = mask_node.output[0]

        if (
            matmul_v.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == root_input
        ):

            attention_last_node = reshape_qkv 
            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            if (q_num_heads == 0 and q_hidden_size == 0):
                q_num_heads, q_hidden_size = self.try_to_get_num_heads_and_hidden_size_for_self_attention(matmul_q)
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            if add_q is None and add_k is None and add_v is None:
                new_node = self.create_matmul_attention_node(
                    mask_index,
                    matmul_q,
                    matmul_k,
                    matmul_v,
                    q_num_heads,
                    q_hidden_size,
                    root_input,
                    attention_last_node.output[0],
                    add_qk_str,
                )
            else:
                new_node = self.create_qkv_attention_node(
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

    def fuse2(self, add_node, input_name_to_nodes, output_name_to_node):

        start_node = add_node 
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["MatMul",  "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (_, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            return
        """
        Match Stable fusion 
                                            |
        Transpose -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """

        v_nodes = self.model.match_parent_path(
            matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 1]
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv,["Softmax", "MatMul"], [0, None])
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return
        matmul_qk = None
        (_,  matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, 0, None]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, 0, None]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        root_input = None
        if  matmul_v.input[0] == matmul_q.input[0] and matmul_k.input[0] == matmul_q.input[0]:
            root_input = matmul_v.input[0]

        if root_input is not None:

            attention_last_node = reshape_qkv 
            q_num_heads = 16
            q_hidden_size = 512
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_qkv_attention_node(
                None,
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
                None,
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
