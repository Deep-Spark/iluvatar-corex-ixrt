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

import numpy as np
from onnx import TensorProto, helper, numpy_helper,NodeProto

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class ReplaceCumSum(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CumSum","CumSum")
        self.prune_graph: bool = False

    def fuse(self, cumsum_node, input_name_to_nodes, output_name_to_node):
        cumsum_node.op_type = "CumSum_IxRT"
        
        
 
 
class ReformatFcBias(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Split","Split")
        self.prune_graph: bool = False
        
    
    def create_add_node(self, inputs, bias_data):
        
        add_node_name = self.model.create_node_name("ReformatFcBias")
        const_data_name = add_node_name+"_constant_data"
        
        
        output_name = add_node_name+"_output"
        bias_weight = helper.make_tensor(const_data_name, TensorProto.FLOAT, bias_data.shape, bias_data)
        self.model.add_initializer(bias_weight)
        
        add_node = helper.make_node(
            "Add",
            inputs = [const_data_name, inputs[0]],
            outputs = [output_name],
            name = add_node_name,
        )
        return add_node    
        
        
    def fuse(self, split_node, input_name_to_nodes, output_name_to_node):
        paths = {"path_0": (["MatMul"],[0])}
        nodes, path_0 = self.match_parent_path_from_dict(
            split_node, paths
        )
        #splot qkv
        
        if nodes is None:
            return
        
        if len(split_node.output) != 3:
            return
        
        
        split_data =  self.model.get_initializer(split_node.input[1])
        if split_data is None:
            return
        
        split_data_np = numpy_helper.to_array(split_data)
        
        if split_data_np.shape != (3,):
            return
        
        if (split_data_np[0] != split_data_np[1]) and (split_data_np[0] != split_data_np[2]):
            return
        
        matmul_weight =  self.model.get_initializer(nodes[0].input[1]) 
        
        if matmul_weight is None:
            return 
        
        matmul_weight_np = numpy_helper.to_array(matmul_weight)
        if matmul_weight_np.shape[1] != 3 * split_data_np[0]:
            return
        
        bias_nodes =  self.model.get_children(split_node)
        
        if bias_nodes is None:
            return
        
        modify = False
        for node in bias_nodes:
            if node.op_type == "Add":
                modify = True
                                 
        if  not  modify:
            return
                
        #q_bias, k_bias, v_bias   
        # q_bias_node,  k_bias_node, v_bias_node  =  bias_nodes[0],bias_nodes[1],bias_nodes[2]
        
        qkv_bias = []
        
        dele_add_nodes = []
        for node in bias_nodes:
            if node.op_type =="Add":
                bias_input0 =  self.model.get_initializer(node.input[0])
                bias_input1 =  self.model.get_initializer(node.input[1])
                
                bias_weight = bias_input0  if  bias_input0  else bias_input1
                                     
                if  bias_weight is None:
                    return
                bias_weight_np = numpy_helper.to_array(bias_weight)
                if 3* bias_weight_np.shape[0] != matmul_weight_np.shape[1]:
                    return
                dele_add_nodes.append(node)
                
            else:
                bias_weight_np = np.zeros((matmul_weight_np.shape[1]//3, ),dtype = matmul_weight_np.dtype) #empty bias
                
                dele_add_nodes.append(None)
                
            qkv_bias.append(bias_weight_np)        
        qkv_bias = np.concatenate(qkv_bias, axis = 0).astype(np.float32) 
        
        add_node_inputs =  nodes[0].output
        new_bias_node = self.create_add_node(add_node_inputs,qkv_bias)
        self.model.replace_input_of_all_nodes(split_node.input[0], new_bias_node.output[0])
        

                
        for index, node in enumerate(dele_add_nodes):
            if node is not None:
                child_node = self.model.get_children(node) 
                self.model.replace_output_of_all_nodes(split_node.output[index], node.output[0])
                  
        dele_nodes = [node for node in dele_add_nodes if node is not None ] 
        
        self.node_name_to_graph_name[new_bias_node.name] = self.this_graph_name
        self.nodes_to_add.append(new_bias_node)
        self.nodes_to_remove.extend(dele_nodes)
        
        


class FusionSplitQKV(Fusion):
    """
    Fuse FusionSplitQKV
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "Split", "Split")
        
    def create_splitqkv_node(
        self, inputs,outputs,num_head,num_dim 
    ) -> Union[NodeProto, None]:
        """Create an XSoftmax node.

        Args:
            data_input (str): data input name
            mask_input (str): max input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        node_name = self.model.create_node_name("SplitQKV_IxRT")

        new_node = helper.make_node(
            "SplitQKV_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=node_name,
        )
        new_node.domain = "com.iluvatar"
        new_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        new_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        new_node.attribute.extend([helper.make_attribute("atten_scale", 1.0)])
        new_node.attribute.extend([helper.make_attribute("transpose", 1)])
        new_node.attribute.extend([helper.make_attribute("num_head", num_head)])
        new_node.attribute.extend([helper.make_attribute("num_dim", num_dim)])
    
        return new_node

    def fuse(self, split_node, input_name_to_nodes, output_name_to_node):
        
        split_data =  self.model.get_initializer(split_node.input[1])
        if split_data is None:
            return
        
        split_data_np = numpy_helper.to_array(split_data)
        
        if split_data_np.shape != (3,):
            return
        
        if (split_data_np[0] != split_data_np[1]) and (split_data_np[0] != split_data_np[2]):
            return
        
        hidden_size = split_data_np[0]
        reshape_nodes =  self.model.get_children(split_node)
                
        if reshape_nodes is None or len(reshape_nodes) != 3:
            return 
        
        for node in reshape_nodes:
            if node.op_type != "Reshape":
                return
            
        q_reshape,k_reshape,v_reshape =  reshape_nodes[0],reshape_nodes[1],reshape_nodes[2] 
        
        q_shape_data = self.model.get_initializer(q_reshape.input[1])  
        k_shape_data = self.model.get_initializer(k_reshape.input[1])  
        v_shape_data = self.model.get_initializer(v_reshape.input[1])  
        if q_shape_data is None or k_shape_data is None or v_shape_data is None:
            return
        
        q_shape_data_np = numpy_helper.to_array(q_shape_data)
        k_shape_data_np = numpy_helper.to_array(k_shape_data)
        v_shape_data_np = numpy_helper.to_array(v_shape_data)
                
        if (q_shape_data_np != k_shape_data_np).any() and (q_shape_data_np != v_shape_data_np).any():
            return
        
        if q_shape_data_np.shape[0] != 4:
            return
        
        num_heads   = q_shape_data_np[2] if q_shape_data_np[2] > 0 else (hidden_size // q_shape_data_np[3])
        head_dims = q_shape_data_np[3] if q_shape_data_np[3] > 0 else (hidden_size // q_shape_data_np[2])
        
        
        q_transpose_node =  self.model.find_first_child_by_type(q_reshape,"Transpose")
        k_transpose_node =  self.model.find_first_child_by_type(k_reshape,"Transpose")
        v_transpose_node =  self.model.find_first_child_by_type(v_reshape,"Transpose")
        
        
        if q_transpose_node is None or  q_transpose_node.attribute[0].ints != [0, 2, 1, 3]:
            return
        
        if k_transpose_node is None or  k_transpose_node.attribute[0].ints != [0, 2, 1, 3]:
            return
        
        if v_transpose_node is None or  v_transpose_node.attribute[0].ints != [0, 2, 1, 3]:
            return
        
        plugin_inputs = [split_node.input[0]]
        plugin_outputs = [q_transpose_node.output[0],k_transpose_node.output[0],v_transpose_node.output[0]]
        
        plugin_node = self.create_splitqkv_node(plugin_inputs,plugin_outputs,num_heads,head_dims)
        
        self.nodes_to_add.append(plugin_node)
        self.node_name_to_graph_name[plugin_node.name] = self.this_graph_name
        
        self.nodes_to_remove.append(split_node)
        
        self.nodes_to_remove.extend(reshape_nodes)
        self.nodes_to_remove.append(q_transpose_node)
        self.nodes_to_remove.append(k_transpose_node)
        self.nodes_to_remove.append(v_transpose_node)
        
        
                
                
                
            

                
  
                    

                
                
                
                
        
        
                
        
        
        

        


        
        
        
        
        
        
        

        
         
           

