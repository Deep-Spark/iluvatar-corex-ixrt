from logging import getLogger
from os import name
from sys import path
from typing import List, Tuple, Union

from onnx import NodeProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionCrossAttention(Fusion):
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
            "CustomQkvCrossToContext_IxRT",
            ["Softmax"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def check_cross_attention(self, softmax_node, input_name_to_nodes, output_name_to_node):
        # the input and output of the softmax node are the matmul node
        softmax_in_node = self.model.get_parents(softmax_node, output_name_to_node)[0]
        if softmax_in_node.op_type != "MatMul":
            return False

        softmax_out_node = self.model.get_children(softmax_node, input_name_to_nodes)[0]
        if softmax_out_node.op_type != "MatMul":
            return False

        # find three matmul
        q_node = self.model.get_parent(softmax_in_node, 0, output_name_to_node)
        k_node = self.model.get_parent(softmax_in_node, 1, output_name_to_node)
        v_node = self.model.get_parent(softmax_out_node, 1, output_name_to_node)

        q_matmul = self.model.find_first_parent_by_type(q_node, "MatMul", output_name_to_node)
        if not q_matmul:
            return False

        k_matmul = self.model.find_first_parent_by_type(k_node, "MatMul", output_name_to_node)
        if not k_matmul:
            return False

        v_matmul = self.model.find_first_parent_by_type(v_node, "MatMul", output_name_to_node)
        if not v_matmul:
            return False

        if q_matmul.input[0] == k_matmul.input[0] or k_matmul.input[0] != v_matmul.input[0]:
            return False

        return True

    def check_num_heads_and_hidden_size(self, qk_node: NodeProto):
        tensor_type = None
        for value_info in self.model.model.graph.value_info:
            if value_info.name == qk_node.input[1]:
                tensor_type = value_info.type.tensor_type
                break
        if tensor_type is None:
            logger.debug(f"{qk_node.input[1]} is not initializer.")
            return False

        shape_value = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape_value.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape_value.append(dim.dim_param)

        if len(shape_value)!= 4:
            logger.debug("shape_value is not 4 dims.")
            return False

        if not (isinstance(shape_value[1], int) and isinstance(shape_value[2], int)):
            logger.debug("shape_value is not int, so skip check num_heads and hidden_size.")
            return True

        if shape_value[1] <= 0 or shape_value[2] <= 0:
            logger.debug(f"shape_value={shape_value}.")
            return False

        num_heads = shape_value[1]
        head_size = shape_value[2]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(
                    f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value."
                )
                self.num_heads_warning = False

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = (
                    False  # Do not show the warning more than once
                )

        return True

    def create_attention_node(self, q_in, k_in, v_in, qkv_out, scale) -> Union[NodeProto, None]:
        attention_node = helper.make_node(
            "CustomQkvCrossToContext_IxRT",
            inputs=[q_in, k_in, v_in],
            outputs=[qkv_out],
            name=self.model.create_node_name("CrossAttention")
        )

        attention_node.domain = "com.iluvatar"
        attention_node.attribute.append(helper.make_attribute("type_id", 1))
        attention_node.attribute.append(helper.make_attribute("has_mask", 0))
        attention_node.attribute.append(helper.make_attribute("type_mask", 3))
        attention_node.attribute.append(helper.make_attribute("layout", 1))
        attention_node.attribute.append(helper.make_attribute("scale", scale))
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.nodes_to_add.append(attention_node)

        return attention_node

    def fuse(self, softmax_node, input_name_to_nodes, output_name_to_node):
        """
        Match CrossAttention pattern of the stable diffusion, the input edge of the Q is different from the input edge of the K and V.

                 --> Transpose --> Reshape --> MatMul ---------------------------------
        MatMul --                                                                      |--> Edge_1
                 --> Softmax --> MatMul --> Mul --> Transpose --> Reshape --> MatMul --
                                    |
                                     --> Mul -> Transpose --> Reshape --> MatMul --> Edge_2
        """
        is_cross_attention = self.check_cross_attention(softmax_node, input_name_to_nodes, output_name_to_node)
        if not is_cross_attention:
            logger.debug("The pattern of CrossAttention is not matched, skip fusion")
            return

        qkv_matmul_node = self.model.get_children(softmax_node, input_name_to_nodes)[0]
        qk_matmul_node = self.model.get_parent(softmax_node, 0, output_name_to_node)
        if not self.check_num_heads_and_hidden_size(qk_matmul_node):
            return

        q_mul_node = self.model.get_parent(qk_matmul_node, 0, output_name_to_node)
        k_mul_node = self.model.get_parent(qk_matmul_node, 1, output_name_to_node)
        if not q_mul_node or not k_mul_node:
            logger.debug("The input of QK MatMul is not Mul, skip fusion")
            return

        q_trans_node = self.model.get_parent(q_mul_node, 0, output_name_to_node)
        k_trans_node = self.model.get_parent(k_mul_node, 0, output_name_to_node)
        v_trans_node = self.model.get_parent(qkv_matmul_node, 1, output_name_to_node)
        qkv_trans_node = self.model.get_children(qkv_matmul_node, input_name_to_nodes)[0]

        q_in = q_trans_node.input[0]
        k_in = k_trans_node.input[0]
        v_in = v_trans_node.input[0]
        qkv_out = qkv_trans_node.output[0]

        q_scale = NumpyHelper.to_array(self.model.get_initializer(q_mul_node.input[1]))[0]
        k_scale = NumpyHelper.to_array(self.model.get_initializer(k_mul_node.input[1]))[0]

        self.create_attention_node(q_in, k_in, v_in, qkv_out, q_scale * k_scale)
        self.nodes_to_remove.extend([qkv_matmul_node, qk_matmul_node, softmax_node])
        self.nodes_to_remove.extend([q_mul_node, k_mul_node])
        self.nodes_to_remove.extend([q_trans_node, k_trans_node, v_trans_node, qkv_trans_node])

        self.prune_graph = True
