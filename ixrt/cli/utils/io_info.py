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

import os.path
import sys
from collections import defaultdict

import ixrt

from .exec_parser import onnx_parser


def convert_onnx_type(onnx_data_type):
    if onnx_data_type == "FLOAT":
        return "float32"
    elif onnx_data_type == "FLOAT16":
        return "float16"
    elif onnx_data_type == "INT8":
        return "int8"
    elif onnx_data_type == "DOUBLE":
        return "float64"
    elif onnx_data_type == "INT32":
        return "int32"
    elif onnx_data_type == "INT64":
        return "int64"
    elif onnx_data_type == "BOOL":
        return "bool"
    elif onnx_data_type == "UINT8":
        return "uint8"
    else:
        raise ValueError(f"unsupported onnx type {onnx_data_type}")


class DynamicDescriptor:
    def __init__(self):
        self.name = ""
        self.min_shape = []
        self.opt_shape = []
        self.max_shape = []

    def check(self):
        assert len(self.min_shape) == len(
            self.opt_shape
        ), "The ranks of the shapes must be consistent."
        assert len(self.opt_shape) == len(
            self.max_shape
        ), "The ranks of the shapes must be consistent."


def parse_inference_input_shapes(exec_config):
    if exec_config.shapes:
        shape_map = {}
        for shape_desc in exec_config.shapes.split(","):
            shape_desc_str = shape_desc.strip()
            last_separate_idx = shape_desc_str.rfind(":")
            name = shape_desc_str[:last_separate_idx]
            shape_string = shape_desc_str[last_separate_idx + 1 :]
            shape = [int(i) for i in shape_string.split("x")]
            shape_map[name] = shape
        return shape_map
    else:
        return {}


def get_io_info(exec_config):
    onnx_model_inputs, onnx_model_outputs = onnx_parser(exec_config.onnx)
    ixrt_input_shapes = {}
    is_dynamic_model = False
    is_dynamic_shape = {}
    for tensor_name, tensor_info in onnx_model_inputs.items():
        input_shape = tensor_info["shape"]
        if len(input_shape) == 0:
            raise ValueError(f"input shape not provided : {tensor_name}")
        is_dynamic_shape[tensor_name] = False
        for shape_val in input_shape:
            if isinstance(shape_val, str) or shape_val == -1:
                is_dynamic_model = True
                is_dynamic_shape[tensor_name] = True
                break
        ixrt_input_shapes[tensor_name] = input_shape
    # Parse dynamic input shapes if possible
    if is_dynamic_model:
        use_dynamic_input = (
            exec_config.min_shape and exec_config.opt_shape and exec_config.max_shape
        )
        assert (
            use_dynamic_input
        ), "If you use dynamic shape input, please specify all --min_shape,--opt_shape,--max_shape"
        ixrt_input_shapes["dynamic_input"] = defaultdict(DynamicDescriptor)
        for shape_map in exec_config.min_shape.split(","):
            last_separate_idx = shape_map.rfind(":")
            input_name = shape_map[:last_separate_idx]
            input_shape = shape_map[last_separate_idx + 1 :]
            assert (
                input_name in ixrt_input_shapes
            ), "Cannot find input tensor with name '{}' in the network inputs! Please make sure the input tensor names are correct.".format(
                input_name
            )
            ixrt_input_shapes["dynamic_input"][input_name].name = input_name
            input_shape = ixrt.Dims(
                [int(shape) for shape in input_shape.split("x")]
            )
            ixrt_input_shapes["dynamic_input"][input_name].min_shape = input_shape
            assert len(ixrt_input_shapes[input_name]) == len(
                input_shape
            ), "The rank of '{}' is {}, please make sure the input tensor shape is correct.".format(
                input_name, len(ixrt_input_shapes[input_name])
            )
        for shape_map in exec_config.opt_shape.split(","):
            last_separate_idx = shape_map.rfind(":")
            input_name = shape_map[:last_separate_idx]
            input_shape = shape_map[last_separate_idx + 1 :]
            assert (
                input_name in ixrt_input_shapes
            ), "Cannot find input tensor with name '{}' in the network inputs! Please make sure the input tensor names are correct.".format(
                input_name
            )
            ixrt_input_shapes["dynamic_input"][input_name].name = input_name
            input_shape = ixrt.Dims(
                [int(shape) for shape in input_shape.split("x")]
            )
            ixrt_input_shapes["dynamic_input"][input_name].opt_shape = input_shape
            assert len(ixrt_input_shapes[input_name]) == len(
                input_shape
            ), "The rank of '{}' is {}, please make sure the input tensor shape is correct.".format(
                input_name, len(ixrt_input_shapes[input_name])
            )
        for shape_map in exec_config.max_shape.split(","):
            last_separate_idx = shape_map.rfind(":")
            input_name = shape_map[:last_separate_idx]
            input_shape = shape_map[last_separate_idx + 1 :]
            assert (
                input_name in ixrt_input_shapes
            ), "Cannot find input tensor with name '{}' in the network inputs! Please make sure the input tensor names are correct.".format(
                input_name
            )
            ixrt_input_shapes["dynamic_input"][input_name].name = input_name
            input_shape = ixrt.Dims(
                [int(shape) for shape in input_shape.split("x")]
            )
            assert len(ixrt_input_shapes[input_name]) == len(
                input_shape
            ), "The rank of '{}' is {}, please make sure the input tensor shape is correct.".format(
                input_name, len(ixrt_input_shapes[input_name])
            )
            ixrt_input_shapes["dynamic_input"][input_name].max_shape = input_shape
        for tensor_name in is_dynamic_shape:
            if is_dynamic_shape[tensor_name]:
                assert (
                    tensor_name in ixrt_input_shapes["dynamic_input"]
                ), "{} is dynamic shape input, please specify all --min_shape,--opt_shape,--max_shape".format(
                    tensor_name
                )
                ixrt_input_shapes["dynamic_input"][tensor_name].check()
    else:
        use_dynamic_input = (
            exec_config.min_shape and exec_config.opt_shape and exec_config.max_shape
        )
        assert (
            not use_dynamic_input
        ), "Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself, please do not pass --min_shape,--opt_shape,--max_shape."

    input_type_dict = {}
    if exec_config.input_types is not None:
        for type_map in exec_config.input_types.split(","):
            last_separate_idx = type_map.rfind(":")
            input_name = type_map[:last_separate_idx]
            input_type = type_map[last_separate_idx + 1 :]
            input_type_dict[input_name] = input_type
    else:
        for tensor_name, tensor_info in onnx_model_inputs.items():
            input_type = convert_onnx_type(tensor_info["type"])
            input_type_dict[tensor_name] = input_type

    output_type_dict = {}
    if exec_config.output_types is not None:
        for type_map in exec_config.output_types.split(","):
            last_separate_idx = type_map.rfind(":")
            output_name = type_map[:last_separate_idx]
            output_type = type_map[last_separate_idx + 1 :]
            output_type_dict[output_name] = output_type
    else:
        for tensor_name, tensor_info in onnx_model_outputs.items():
            output_type = convert_onnx_type(tensor_info["type"])
            output_type_dict[tensor_name] = output_type
    return ixrt_input_shapes, is_dynamic_model, input_type_dict, output_type_dict


def parse_custom_inputs(exec_config):
    inputs = exec_config.load_inputs
    result = defaultdict(str)
    if not inputs:
        return result
    for i in inputs.split(","):
        assert (
            i.find(":") > -1
        ), "--load_inputs should pass name:file for each input entry"
        last_separate_idx = i.rfind(":")
        input_name = i[:last_separate_idx]
        input_file = i[last_separate_idx + 1 :]
        assert os.path.exists(input_file), f"{input_file} not exists!"
        result[input_name] = input_file
    return result
