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

import argparse
import json
import os

import ixrt

TRT_LOGGER = ixrt.Logger(ixrt.Logger.VERBOSE)

EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    return val * 1 << 30


def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def setDynamicRange(network, json_file):
    """Sets ranges for network layers."""
    quant_param_json = json_load(json_file)
    act_quant = quant_param_json["act_quant_info"]

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            print(input_tensor.name)
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = -abs(value)
            input_tensor.dynamic_range = (tensor_min, tensor_max)

    for i in range(network.num_layers):
        layer = network.get_layer(i)

        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)

            if act_quant.__contains__(tensor.name):
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = (tensor_min, tensor_max)
            else:
                print("\033[1;32m%s\033[0m" % tensor.name)


def build_engine(onnx_file, json_file, engine_file):
    builder = ixrt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)

    config = builder.create_builder_config()

    # If it is a dynamic onnx model , you need to add the following.
    # profile = builder.create_optimization_profile()
    # profile.set_shape("input_name", (batch, channels, min_h, min_w), (batch, channels, opt_h, opt_w), (batch, channels, max_h, max_w))
    # config.add_optimization_profile(profile)

    parser = ixrt.OnnxParser(network, TRT_LOGGER)
    # config.max_workspace_size = GiB(1)

    if not os.path.exists(onnx_file):
        quit("ONNX file {} not found".format(onnx_file))

    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config.set_flag(ixrt.BuilderFlag.INT8)

    setDynamicRange(network, json_file)

    engine = builder.build_engine(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    # Add plugins if needed
    # import ctypes
    # ctypes.CDLL("libmmdeploy_ixrt_ops.so")
    parser = argparse.ArgumentParser(
        description="Writing qparams to onnx to convert ixrt engine."
    )
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--qparam_json", type=str, default=None)
    parser.add_argument("--engine", type=str, default=None)
    arg = parser.parse_args()

    build_engine(arg.onnx, arg.qparam_json, arg.engine)
    print("\033[1;32mgenerate %s\033[0m" % arg.engine)
