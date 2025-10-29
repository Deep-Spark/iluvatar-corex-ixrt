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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import struct
import time

import cuda.cudart as cudart
import cv2
import numpy as np
from imagenet_labels import labels

import ixrt
from ixrt.utils import topk


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="api_to_model",
        help="model name",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../../../data/resnet18/resnet18_fusebn.wts"
        ),
        help="weight file path",
    )

    config = parser.parse_args()
    config.image_file = os.path.join(
        os.path.dirname(config.weight_path), "kitten_224.bmp"
    )
    return config


def show_cls_result(result, k=5):
    data = result

    vals, idxs = topk(data, k, axis=1)
    idx0 = idxs[0]
    val0 = vals[0]
    print(
        "------------------------------Python inference result------------------------------"
    )
    for i, (val, idx) in enumerate(zip(val0, idx0)):
        print(f"Top {i+1}:   {val}  {labels[idx]}")


def load_weights(file):
    assert os.path.exists(file), f"Unable to load weight file {file}"

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def basic_block(
    network, weight_map, input, in_channels, out_channels, stride, layer_name
):

    conv1 = network.add_convolution_nd(
        input=input,
        num_output_maps=out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[layer_name + "conv1.weight"],
        bias=weight_map[layer_name + "conv1.bias"],
    )
    assert conv1
    conv1.stride_nd = (stride, stride)
    conv1.padding_nd = (1, 1)

    relu1 = network.add_activation(conv1.get_output(0), type=ixrt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution_nd(
        input=relu1.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[layer_name + "conv2.weight"],
        bias=weight_map[layer_name + "conv2.bias"],
    )
    assert conv2
    conv2.padding_nd = (1, 1)

    if in_channels != out_channels:
        conv3 = network.add_convolution_nd(
            input=input,
            num_output_maps=out_channels,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=weight_map[layer_name + "downsample.0.bias"],
        )
        assert conv3
        conv3.stride_nd = (stride, stride)

        ew1 = network.add_elementwise(
            conv3.get_output(0), conv2.get_output(0), ixrt.ElementWiseOperation.SUM
        )
    else:
        ew1 = network.add_elementwise(
            input, conv2.get_output(0), ixrt.ElementWiseOperation.SUM
        )
    assert ew1

    relu2 = network.add_activation(ew1.get_output(0), type=ixrt.ActivationType.RELU)
    assert relu2

    return relu2


def construct_network(network, weight):
    model_info = {
        "input_name": "input",
        "input_shape": [1, 3, 224, 224],
        "output_name": "output",
        "dtype": ixrt.float32,
    }
    data = network.add_input(
        name="input", dtype=model_info["dtype"], shape=model_info["input_shape"]
    )

    conv1 = network.add_convolution_nd(
        input=data,
        num_output_maps=64,
        kernel_shape=(7, 7),
        kernel=weight["conv1.weight"],
        bias=weight["conv1.bias"],
    )
    assert conv1
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (3, 3)

    relu1 = network.add_activation(conv1.get_output(0), type=ixrt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling_nd(
        input=relu1.get_output(0), window_size=(3, 3), type=ixrt.PoolingType.MAX
    )
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (1, 1)

    relu2 = basic_block(network, weight, pool1.get_output(0), 64, 64, 1, "layer1.0.")
    relu3 = basic_block(network, weight, relu2.get_output(0), 64, 64, 1, "layer1.1.")
    relu4 = basic_block(network, weight, relu3.get_output(0), 64, 128, 2, "layer2.0.")
    relu5 = basic_block(network, weight, relu4.get_output(0), 128, 128, 1, "layer2.1.")
    relu6 = basic_block(network, weight, relu5.get_output(0), 128, 256, 2, "layer3.0.")
    relu7 = basic_block(network, weight, relu6.get_output(0), 256, 256, 1, "layer3.1.")
    relu8 = basic_block(network, weight, relu7.get_output(0), 256, 512, 2, "layer4.0.")
    relu9 = basic_block(network, weight, relu8.get_output(0), 512, 512, 1, "layer4.1.")

    pool2 = network.add_pooling_nd(
        input=relu9.get_output(0), window_size=(7, 7), type=ixrt.PoolingType.AVERAGE
    )
    assert pool2
    pool2.stride_nd = (1, 1)

    flatten = network.add_shuffle(input=pool2.get_output(0))
    assert flatten
    flatten.reshape_dims = (1, 512)

    gemm_weight = network.add_constant(shape=(1000, 512), weights=weight["fc.weight"])
    matmul = network.add_matrix_multiply(
        input0=flatten.get_output(0),
        op0=ixrt.MatrixOperation.NONE,
        input1=gemm_weight.get_output(0),
        op1=ixrt.MatrixOperation.TRANSPOSE,
    )
    assert matmul

    gemm_bias = network.add_constant(shape=(1, 1000), weights=weight["fc.bias"])
    gemm_bias_add = network.add_elementwise(
        matmul.get_output(0), gemm_bias.get_output(0), ixrt.ElementWiseOperation.SUM
    )
    assert gemm_bias_add

    network.mark_output(tensor=gemm_bias_add.get_output(0))


def build(config):
    print("====================build====================")
    IXRT_LOGGER = ixrt.Logger(ixrt.Logger.WARNING)
    builder = ixrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    weight = load_weights(config.weight_path)
    # construct layer by layer
    construct_network(network, weight)

    build_config.set_flag(ixrt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = os.path.join(
        os.path.dirname(config.weight_path), config.model_name + "_fp16.engine"
    )
    with open(engine_file_path, "wb") as f:
        f.write(plan)

    print("engine_file_path: ", engine_file_path)
    print("Build engine done!")


def execute(config):
    print("====================execute====================")
    engine_path = os.path.join(
        os.path.dirname(config.weight_path), config.model_name + "_fp16.engine"
    )

    logger = ixrt.Logger(ixrt.Logger.ERROR)
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

        # Setup I/O bindings
        inputs = []
        outputs = []
        allocations = []
        for i in range(engine.num_bindings):
            name = engine.get_tensor_name(i)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)

            size = np.dtype(ixrt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            err, allocation = cudart.cudaMalloc(size)
            assert err == cudart.cudaError_t.cudaSuccess
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(ixrt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "nbytes": size,
            }
            allocations.append(allocation)
            if engine.get_tensor_mode(name) == ixrt.TensorIOMode.INPUT:
                inputs.append(binding)
            else:
                outputs.append(binding)
            context.set_tensor_address(engine.get_tensor_name(i), allocation)

        # Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

        data_batch = (
            np.flip(cv2.imread(config.image_file) / 255.0, axis=2)
            .astype("float32")
            .transpose(2, 0, 1)
        )
        data_batch = data_batch.reshape(1, *data_batch.shape)
        data_batch = np.ascontiguousarray(data_batch)

        # Process I/O and execute the network
        assert inputs[0]["nbytes"] == data_batch.nbytes
        (err,) = cudart.cudaMemcpy(
            inputs[0]["allocation"],
            data_batch,
            inputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess
        context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)

        assert outputs[0]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output,
            outputs[0]["allocation"],
            outputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        (err,) = cudart.cudaFree(inputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess

        show_cls_result(output)


if __name__ == "__main__":
    config = parse_config()

    build(config)
    execute(config)
