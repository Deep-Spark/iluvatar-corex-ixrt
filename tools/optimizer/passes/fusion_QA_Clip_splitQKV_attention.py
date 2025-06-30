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
from typing import Tuple, Union

from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel
import numpy as np
logger = getLogger(__name__)


class FusionSplitQKVWithAtten(Fusion):
    """
    Fuse FusionSplitQKV
    """

    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int):
        super().__init__(model, "SplitQKV", "Softmax")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def create_node(
        self, inputs: list, outputs:list,input_reshape_data,output_reshape_data,num_head,head_dim,scale
    ) -> Union[NodeProto, None]:
        """Create an create node.

        Args:
            data_input (str): data input name
            mask_input (str): max input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        node_name = self.model.create_node_name("transformer")
        
        # input_reshape_name = node_name+"_plugin_input_reshape"
        # input_reshape_shape_data_name = input_reshape_name+"_shape_data"
        # input_reshape_output_name = input_reshape_name + "_output"
        
        # input_reshape_node = helper.make_node(
        #     "Reshape",
        #     inputs = [inputs[0],input_reshape_shape_data_name],
        #     outputs = [input_reshape_output_name],
        #     name = input_reshape_name,
        #     allowzero = 0
        # )
        
        # input_reshape_shape_tensor = helper.make_tensor(input_reshape_shape_data_name, TensorProto.INT64, input_reshape_data.shape, input_reshape_data)
        # self.model.add_initializer(input_reshape_shape_tensor)
        
        splitqkv_node_name = node_name+"_plugin_splitQKV"
        q_out_name = splitqkv_node_name +"_query"
        k_out_name = splitqkv_node_name +"_key"
        v_out_name = splitqkv_node_name +"_value"
        
    
        splitqkv_node = helper.make_node(
            "SplitQKV_IxRT",
            inputs=[inputs[0]],
            outputs=[q_out_name,k_out_name,v_out_name],
            name=splitqkv_node_name,
        )
        splitqkv_node.domain = "com.iluvatar"
        splitqkv_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        splitqkv_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        splitqkv_node.attribute.extend(
            [helper.make_attribute("atten_scale", 1.0)]
        )
        splitqkv_node.attribute.extend(
            [helper.make_attribute("transpose", 2)]
        )
        splitqkv_node.attribute.extend([helper.make_attribute("num_head", num_head)])
        splitqkv_node.attribute.extend(
            [helper.make_attribute("head_dim", head_dim)])
        
        
    
    
        attention_node_name = node_name+"_plugin_attention"
        attention_node_outname = attention_node_name + "_output"
        
        attention_node = helper.make_node(
            "CustomQkvCrossToContext_IxRT",
            inputs=splitqkv_node.output,
            outputs=[attention_node_outname],
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 1)])
        attention_node.attribute.extend([helper.make_attribute("scale", scale)])
        attention_node.attribute.extend([helper.make_attribute("has_mask", 0)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("type_mask", 3)])
        
        
        
        
        output_reshape_name = node_name+"_plugin_output_reshape"
        output_reshape_shape_data_name = output_reshape_name+"_shape_data"

        
        output_reshape_node = helper.make_node(
            "Reshape",
            inputs = [attention_node.output[0],output_reshape_shape_data_name],
            outputs = outputs,
            name = output_reshape_name,
            allowzero = 0
        )
        
        output_reshape_shape_tensor = helper.make_tensor(output_reshape_shape_data_name, TensorProto.INT64, output_reshape_data.shape, output_reshape_data)
        self.model.add_initializer(output_reshape_shape_tensor)
        
        return splitqkv_node,attention_node,output_reshape_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        
        
        
        
        
        """
         path1:

         (query) --Slice----Reshape----Transpose-------Div--->MatMul---->softmax --->MatMul---->
                                                                /                      /
         (key)   ----Slice----Reshape----Transpose --------- > /                      /
                                                                                    /
                                                                                   /
         (value)--------Slice-----Reshape------Transpose------------------------->
        """

        #print(node)
        q_paths = {
            "q_path": (
                ["MatMul", "Div", "Transpose", "Reshape","Slice"],
                [ None,     None,    None,   None,    None],
                
            ),  
        }
        
        q_path_nodes, q_path = self.match_parent_path_from_dict(
            node, q_paths
        )
        if q_path_nodes is None:
            return
        
        
        k_paths = {
            "k_path": (
                ["MatMul",  "Transpose", "Reshape","Slice"],
                [ None,      None,   None,    None],
                
            ),  
        }
        
        
        k_path_nodes, k_path = self.match_parent_path_from_dict(
            node, k_paths
        )
        
        
        if k_path_nodes is None:
            return
        
        
        next_nodes = self.model.get_children(node)
    
        if len(next_nodes) == 0:
            return
                
        if next_nodes[0].op_type != "MatMul":
            return
        
        second_matmul_node = next_nodes[0]
        
        
        
        v_paths = {
            "v_path": (
                ["Transpose", "Reshape","Slice"],
                [ None,   None,    None],
                
            ),  
        }
        
        
        v_path_nodes, k_path = self.match_parent_path_from_dict(
            second_matmul_node, v_paths
        )
        
        slice_node_q = q_path_nodes[-1]
        slice_node_k = k_path_nodes[-1]
        slice_node_v = v_path_nodes[-1]
        
        
        if slice_node_q.input[0] != slice_node_k.input[0] and slice_node_q.input[0] != slice_node_v.input[0] :  
            return
        
        
        slice_end_data = numpy_helper.to_array(self.model.get_initializer(slice_node_q.input[2]))
        
        if slice_end_data.shape != (1,):
            return
        
        hidden_size = slice_end_data[0]
        
        reshape_node_q = q_path_nodes[-2]
        reshape_node_k = k_path_nodes[-2]
        reshape_node_v = v_path_nodes[-2]
        
        reshape_data_q = numpy_helper.to_array(self.model.get_initializer(reshape_node_q.input[1]))
        reshape_data_k = numpy_helper.to_array(self.model.get_initializer(reshape_node_k.input[1]))
        reshape_data_v = numpy_helper.to_array(self.model.get_initializer(reshape_node_v.input[1]))
        
        if reshape_data_q.shape != (3,) or reshape_data_k.shape != (3,) or reshape_data_v.shape != (3,) :
            return
        
        sequne_len = reshape_data_q[0]
        head_dim = reshape_data_q[2]
        num_head = hidden_size // head_dim
        
        if  sequne_len==0 or sequne_len==-1:
            return
        
        
        input_reshape_data = np.array([-1, sequne_len, 3* hidden_size ],np.int64) #[sequne_len, bsz, 3* head_size] ->[bsz, sequne_len,3* head_size]
        output_reshape_data = np.array([-1, sequne_len, head_dim ],np.int64) #[bsz, num_head, sequne_len, head_dim] ->[bsz*num_head, sequne_len, head_dim]
        
        split_qkv_inputs = [slice_node_q.input[0]]
        atten_node_outputs = second_matmul_node.output
        
        div_node = q_path_nodes[-4]
        scale = 1.0 / numpy_helper.to_array(self.model.get_initializer(div_node.input[1]))
        splitqkv_node,attention_node,output_reshape_node = self.create_node(split_qkv_inputs,atten_node_outputs,input_reshape_data,output_reshape_data,num_head,head_dim,scale)
        self.nodes_to_add.append(splitqkv_node)
        self.nodes_to_add.append(attention_node)
        self.nodes_to_add.append(output_reshape_node)
        self.node_name_to_graph_name[splitqkv_node.name] = self.this_graph_name
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[output_reshape_node.name] = self.this_graph_name
        self.nodes_to_remove.extend(q_path_nodes)
        self.nodes_to_remove.extend(k_path_nodes)
        self.nodes_to_remove.extend(v_path_nodes)
        self.nodes_to_remove.append(second_matmul_node)
        
        
        
        

        
        
        
        
        
        
        
        
        
            
        
        
        
        
        # print(v_path_nodes)
        
        # import pdb
        # pdb.set_trace()
    
        """    
        split_node = node
        split_data = self.model.get_initializer_input_edges(node.name,return_np_array = True)
        if split_data[0].shape != (3,):
            return 
        if split_data[0][0] != split_data[0][1] and  split_data[0][1] != split_data[0][2]:
            return

        q_input, k_input, v_input = node.output[0],node.output[1],node.output[2]  
              
        q_path_nodes= []
        k_path_nodes= []
        v_path_nodes= []
        
        reshape_nodes = self.model.get_children(node)
        
        for node in reshape_nodes:
            if node.op_type != "Reshape":
                return
        q_reshape_node,k_reshape_node,v_reshape_node =  reshape_nodes[0],reshape_nodes[1],reshape_nodes[2]   
                    
        q_path_nodes.append(q_reshape_node)
        k_path_nodes.append(k_reshape_node)    
        v_path_nodes.append(v_reshape_node) 
        
        q_transpose_nodes = self.model.get_children(q_reshape_node) 
        k_transpose_nodes = self.model.get_children(k_reshape_node) 
        v_transpose_nodes = self.model.get_children(v_reshape_node)
        
        if  len(q_transpose_nodes)!=1 and  (not k_transpose_nodes) and len(v_transpose_nodes) != 1:
            return
        
        
        if (q_transpose_nodes[0].attribute[0].ints != [0, 2, 1, 3]) and (v_transpose_nodes[0].attribute[0].ints !=[0, 2, 1, 3]):
                return 
        
        if len(k_transpose_nodes) == 2:
            if (k_transpose_nodes[0].attribute[0].ints != k_transpose_nodes[1].attribute[0].ints) and (k_transpose_nodes[0].attribute[0].ints !=[0, 2, 1, 3]):
                return 
            
        
        if len(k_transpose_nodes) == 1:
            if  (k_transpose_nodes[0].attribute[0].ints !=[0, 2, 1, 3]):
                return 
                
        
        q_transpose_node = q_transpose_nodes[0]
        k_transpose_node_0 = k_transpose_nodes[0]
        v_transpose_node = v_transpose_nodes[0]
        
        k_output = k_transpose_node_0.output[0]
        
        if len(k_transpose_nodes) == 2:
            k_transpose_node_1 = k_transpose_nodes[1]
            next_node = self.model.get_children(k_transpose_node_1)
            if not next_node:
                return
                        
            self.model.replace_node_input(next_node[0], k_transpose_node_1.output[0], k_transpose_node_0.output[0])
            

        q_path_nodes.append(q_transpose_node)
        v_path_nodes.append(v_transpose_node)
        k_path_nodes.extend(k_transpose_nodes)
        
        plugin_inputs = [split_node.input[0]] 
        plugin_outputs = [q_transpose_node.output[0], k_output,v_transpose_node.output[0]]
        
        remove_nodes = [split_node]
        
        remove_nodes.extend(q_path_nodes)
        remove_nodes.extend(k_path_nodes)
        remove_nodes.extend(v_path_nodes)
                
        new_node,k_cache_concat_node, v_cache_concat_node = self.create_node(plugin_inputs, plugin_outputs)
        
        self.nodes_to_add.append(new_node)
        self.nodes_to_add.append(k_cache_concat_node)
        self.nodes_to_add.append(v_cache_concat_node)
        
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.node_name_to_graph_name[k_cache_concat_node.name] = self.this_graph_name
        self.node_name_to_graph_name[v_cache_concat_node.name] = self.this_graph_name
        self.nodes_to_remove.extend(remove_nodes)
        """
      
    
