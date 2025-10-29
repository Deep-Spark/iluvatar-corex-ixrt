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
import math
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_options import AttentionMaskFormat
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel
from .shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto
from typing import Tuple, Union,List

logger = getLogger(__name__)


class FusionMultiscaleDeformableAttn(Fusion):
    """
    Fuse VideoBertAttention subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "ReduceSum","ReduceSum")


    def create_disentangled_attention_node(
        self,
        inputs: List[str],
        outputs: List[str],
        spatial_shapes_data,
        level_start_index_data
        
    ) -> Union[NodeProto, None]:
        
        attention_node_name = self.model.create_node_name(
            "MultiscaleDeformableAttnPlugin_TRT"
        )
                
        spatial_shapes_name = attention_node_name+"_"+ "spatial_shapes"
        level_start_index_name  = attention_node_name+"_"+ "level_start_index"
        
        spatial_shapes_weight = helper.make_tensor(spatial_shapes_name, TensorProto.INT64, spatial_shapes_data.shape, spatial_shapes_data)
        level_start_index_weight = helper.make_tensor(level_start_index_name, TensorProto.INT64, level_start_index_data.shape, level_start_index_data)
        plugin_inputs = [inputs[0],spatial_shapes_name,level_start_index_name,inputs[1],inputs[2]]
        
        
        self.model.add_initializer(spatial_shapes_weight)
        self.model.add_initializer(level_start_index_weight)
        
        disentangled_attention_node = helper.make_node(
            "MultiscaleDeformableAttnPlugin_TRT",
            inputs=plugin_inputs,
            outputs=outputs,
            name=attention_node_name,
            im2col_step = 64
        )
        
        return disentangled_attention_node
    
    
    def create_reshape_node(self, inputs, outputs, shape_data):
        
        reshape_node_name = self.model.create_node_name("MultiscaleDeformableAttnPlugin_TRT_output_reshape")
        shape_data_name = reshape_node_name+"_shape_data"
        shape_weight = helper.make_tensor(shape_data_name, TensorProto.INT64, shape_data.shape, shape_data)
        self.model.add_initializer(shape_weight)
        reshape_node = helper.make_node(
            "Reshape",
            inputs = [inputs[0],shape_data_name],
            outputs = outputs,
            name = reshape_node_name,
            allowzero = 0
        )
        return reshape_node
        
        
        


    def fuse(self, reduce_sum_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = reduce_sum_node

        value_paths_0 = {
            "value_path_0": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Reshape","Transpose","Gather","Sub","Mul"],
                [None,   None,     None,      0,          None,       1,    None,       None,    None,  None],
            ),
        }
        
        value_nodes_0, value_path_0 = self.match_parent_path_from_dict(
            start_node, value_paths_0
        )
         
        if value_nodes_0 is None:
            return
        
        value_paths_1 = {
            "value_path_1": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Reshape","Transpose","Gather","Sub","Mul"],
                [None,   None,     None,      1,          None,       1,    None,       None,    None, None],
            ),
        }
                
        value_nodes_1, value_path_1 = self.match_parent_path_from_dict(
            start_node, value_paths_1
        )
                
        if value_nodes_1 is None:
            return
        
        
        value_paths_2 = {
            "value_path_1": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Reshape","Transpose","Gather","Sub","Mul"],
                [None,   None,     None,      2,          None,       1,    None,       None,    None, None],
            ),
        }
                
        value_nodes_2, value_path_2 = self.match_parent_path_from_dict(
            start_node, value_paths_2
        )
        
        if value_nodes_2 is None:
            return
        
        
        value_paths_3 = {
            "value_path_1": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Reshape","Transpose","Gather","Sub","Mul"],
                [None,   None,     None,      3,          None,       1,    None,       None,    None, None],
            ),
        }
       
        value_nodes_3, value_path_3 = self.match_parent_path_from_dict(
            start_node, value_paths_3
        )
        
        if value_nodes_3 is None:
            return
        
        
        locations_paths_0 = {
            "location_path_0": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Cast","Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      0,           None,         0,  None,     None,       None,     None],
            ),
            "location_path_0_nocast": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample", "Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      0,           None,         None,     None,       None,     None],
            ),
        }
        
        
        location_nodes_0, locations_path_0 = self.match_parent_path_from_dict(
            start_node, locations_paths_0
        )
        
        if location_nodes_0 is None:
            return

        location_0_shape_index = 6
        if locations_path_0 == "location_path_0_nocast":
            location_0_shape_index = 5
        
        locations_paths_1 = {
            "location_path_1": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Cast","Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      1,           None,         0,  None,     None,       None,     None],
            ),
            "location_path_1_nocast": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample", "Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      1,           None,         None,     None,       None,     None],
            ),
        }
        
        location_nodes_1, locations_path_1 = self.match_parent_path_from_dict(
            start_node, locations_paths_1
        )
        
        if location_nodes_1 is None:
            return

        location_1_shape_index = 6
        if locations_path_1 == "location_path_1_nocast":
            location_1_shape_index = 5

        locations_paths_2 = {
            "location_path_2": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Cast","Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      2,           None,         0,  None,     None,       None,     None],
            ),
            "location_path_2_nocast": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample", "Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      2,           None,        None,     None,       None,     None],
            ),
        }
        


        location_nodes_2, locations_path_2 = self.match_parent_path_from_dict(
            start_node, locations_paths_2
        )
        
        if location_nodes_2 is None:
            return

        location_2_shape_index = 6
        if locations_path_2 == "location_path_2_nocast":
            location_2_shape_index = 5
        
        locations_paths_3 = {
            "location_path_3": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample","Cast","Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      3,           None,         0,  None,     None,       None,     None],
            ),
            "location_path_3_nocast": (
                ["Mul", "Reshape", "Concat", "Unsqueeze", "GridSample", "Reshape","Transpose","Reshape","Split"],
                [None,   None,     None,      3,           None,        None,     None,       None,     None],
            ),
        }
        
        location_nodes_3, locations_path_3 = self.match_parent_path_from_dict(
            start_node, locations_paths_3
        )

        if location_nodes_3 is None:
            return

        location_3_shape_index = 6
        if locations_path_3 == "location_path_3_nocast":
            location_3_shape_index = 5
        

            
        weights_paths = {
            "path2": (
                ["Mul", "Reshape", "Transpose"],
                [0, 1, 0],
            ),
        }

        weight_nodes, weights_path = self.match_parent_path_from_dict(
            start_node, weights_paths
        )
        
        if weight_nodes is None:
            return
        
        spatial_shapes_0 = numpy_helper.to_array(self.model.get_initializer(location_nodes_0[location_0_shape_index].input[1]))
        spatial_shapes_1 = numpy_helper.to_array(self.model.get_initializer(location_nodes_1[location_1_shape_index].input[1]))
        spatial_shapes_2 = numpy_helper.to_array(self.model.get_initializer(location_nodes_2[location_2_shape_index].input[1]))
        spatial_shapes_3 = numpy_helper.to_array(self.model.get_initializer(location_nodes_3[location_3_shape_index].input[1]))
        
        if (spatial_shapes_0 == spatial_shapes_1).all() and (spatial_shapes_0 == spatial_shapes_2).all() and (spatial_shapes_0 == spatial_shapes_3).all():
            return
        
        if (len(spatial_shapes_0.shape) != len(spatial_shapes_1.shape)) and (len(spatial_shapes_0.shape) != len(spatial_shapes_2.shape)) and (len(spatial_shapes_0.shape) != len(spatial_shapes_3.shape)) and (len(spatial_shapes_0.shape) != 4):
            return
        
        spatial_shapes_all = np.concatenate((np.expand_dims(spatial_shapes_0[2:],axis = 0),
                                               np.expand_dims(spatial_shapes_1[2:],axis = 0),
                                               np.expand_dims(spatial_shapes_2[2:],axis = 0),
                                               np.expand_dims(spatial_shapes_3[2:],axis = 0)),axis = 0)
        
        level_start_index_0 = 0
        level_start_index_1 = spatial_shapes_0[2]*spatial_shapes_0[3] + level_start_index_0
        level_start_index_2 = spatial_shapes_1[2]*spatial_shapes_1[3] + level_start_index_1
        level_start_index_3 = spatial_shapes_2[2]*spatial_shapes_2[3] + level_start_index_2    
        level_start_index_all = np.array([level_start_index_0,level_start_index_1,level_start_index_2,level_start_index_3],np.int64)
        
        logger.info(f"FusionMultiscaleDeformableAttn   spatial_shapes data {spatial_shapes_all} ,if you change input shape, please check the data")
        logger.info(f"FusionMultiscaleDeformableAttn   level_start_index data {level_start_index_all} if you change input shape, please check the data")
        
                
        value_input_node = value_nodes_0[-1]
        location_input_node = location_nodes_0[-1]
        weight_input_node = weight_nodes[-1]
        
        if  weight_input_node.attribute[0].ints != [0, 2, 1, 3, 4]:
            return 
        
        
        next_nodes_0 = self.model.get_children(start_node)
        if len(next_nodes_0) == 0 or next_nodes_0[0].op_type != "Reshape":
            return
        reshape_node = next_nodes_0[0]
        
    
        next_nodes_1 = self.model.get_children(next_nodes_0[0])
        
        if len(next_nodes_1) == 0 or next_nodes_1[0].op_type != "Transpose":
            return
        
        transpose_node = next_nodes_1[0]
        
        if transpose_node.attribute[0].ints != [0, 2, 1]:
            return
        
        plugin_output_node = start_node
        plugin_input_names = [location_input_node.input[0],value_input_node.input[0],weight_input_node.input[0]]
        plugin_output_names = [start_node.output[0]]      
        remove_nodes = []
        remove_nodes.extend(value_nodes_0)
        remove_nodes.extend(value_nodes_1)
        remove_nodes.extend(value_nodes_2)
        remove_nodes.extend(value_nodes_3)
        remove_nodes.extend(location_nodes_0)
        remove_nodes.extend(location_nodes_1)
        remove_nodes.extend(location_nodes_2)
        remove_nodes.extend(location_nodes_3)
        
        remove_nodes.append(start_node)
        remove_nodes.append(transpose_node)
        remove_nodes.append(reshape_node)
        
        
        #modify reshape data
        
        shape_data_tensor = self.model.get_initializer(reshape_node.input[1])
        
        shape_data = numpy_helper.to_array(shape_data_tensor)
            
        shape_data_copy = shape_data.copy()
        shape_data_copy[1],shape_data_copy[2] = shape_data_copy[2],shape_data_copy[1]
                
        attention_node = self.create_disentangled_attention_node(plugin_input_names,plugin_output_names,spatial_shapes_all,level_start_index_all)
        
        reshape_node = self.create_reshape_node(attention_node.output, transpose_node.output, shape_data_copy)
        
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
        self.nodes_to_add.append(attention_node)
        self.nodes_to_add.append(reshape_node)
        
        self.nodes_to_remove.extend(remove_nodes)

