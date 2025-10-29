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
"""
In this script, we demonstrate how to use IxRT to infer a ResNet50.
Two examples are showed:
    - test_resnet50
    - test_resnet50_new_api

The first one shows a detailed version while the second one hide the
configuration elegantly. We recommend you use the function
`test_resnet50_nice_api` to infer, while just have a look of
`test_resnet50` that how details are implemented, for your information.
"""
import argparse
import json
import os
import re
import time

import cuda.cudart as cudart
import cv2
import numpy as np
import torch
from utils.imagenet_labels import labels as imagenet_labels

import ixrt
from ixrt import Dims
from ixrt.utils import topk

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def setup_io_bindings(engine, context):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []
    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
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
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(ROOT, "data/resnet18"),
        help="graph file path",
    )
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--multicontext", action="store_true")
    parser.add_argument("--dynamicshape", action="store_true")
    parser.add_argument("--dynamicshape_multicontext", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "int8"],
        default="int8",
        help="The precision of datatype",
    )
    config = parser.parse_args()
    config.image_file = os.path.join(config.model_path, "kitten_224.bmp")
    return config


def show_cls_result(result, k=5):
    data = result
    from imagenet_labels import labels

    vals, idxs = topk(data, k, axis=1)
    idx0 = idxs[0]
    val0 = vals[0]
    print(
        "------------------------------Python inference result------------------------------"
    )
    for i, (val, idx) in enumerate(zip(val0, idx0)):
        print(f"Top {i+1}:   {val}  {labels[idx]}")


def test_build_engine_trtapi(config):
    onnx_model = os.path.join(config.model_path, "resnet18.onnx")
    IXRT_LOGGER = ixrt.Logger(ixrt.Logger.WARNING)
    builder = ixrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = ixrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model)

    if config.precision == "int8":
        build_config.set_flag(ixrt.BuilderFlag.INT8)
    else:
        build_config.set_flag(ixrt.BuilderFlag.FP16)

    plan = builder.build_serialized_network(network, build_config)
    if config.precision == "int8":
        engine_file_path = os.path.join(config.model_path, "resnet18_trt_int8.engine")
    else:
        engine_file_path = os.path.join(config.model_path, "resnet18_trt_fp16.engine")
    with open(engine_file_path, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def test_resnet18_trtapi_ixrt(config):
    print("====================test_resnet18_trtapi_ixrt====================")
    if config.precision == "int8":
        engine_path = os.path.join(config.model_path, "resnet18_trt_int8.engine")
    else:
        engine_path = os.path.join(config.model_path, "resnet18_trt_fp16.engine")
    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.ERROR)
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

        # Setup I/O bindings
        inputs, outputs, allocations = setup_io_bindings(engine, context)

        ### infer
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
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context.execute_async_v2(allocations, stream)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context.execute_v2(allocations)
        assert outputs[0]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output,
            outputs[0]["allocation"],
            outputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        # Free Gpu Memory
        (err,) = cudart.cudaFree(inputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess

        show_cls_result(output)


def test_resnet18_trtapi_multicontext(config):
    print("====================test_resnet18_trtapi_multicontext====================")
    if config.precision == "int8":
        engine_path = os.path.join(config.model_path, "resnet18_trt_int8.engine")
    else:
        engine_path = os.path.join(config.model_path, "resnet18_trt_fp16.engine")
    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.ERROR)
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context
        context_2 = engine.create_execution_context()
        assert context_2

        # Setup I/O bindings
        inputs, outputs, allocations = setup_io_bindings(engine, context)
        inputs_2, outputs_2, allocations_2 = setup_io_bindings(engine, context)

        ### infer
        # Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        output_2 = np.zeros(outputs_2[0]["shape"], outputs_2[0]["dtype"])

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
        assert inputs_2[0]["nbytes"] == data_batch.nbytes
        (err,) = cudart.cudaMemcpy(
            inputs_2[0]["allocation"],
            data_batch,
            inputs_2[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            err, stream2 = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context.execute_async_v2(allocations, stream)
            context_2.execute_async_v2(allocations_2, stream2)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamSynchronize(stream2)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream2)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context.execute_v2(allocations)
            context_2.execute_v2(allocations_2)
        assert outputs[0]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output,
            outputs[0]["allocation"],
            outputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        assert outputs_2[0]["nbytes"] == output_2.nbytes
        (err,) = cudart.cudaMemcpy(
            output_2,
            outputs_2[0]["allocation"],
            outputs_2[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        # Free Gpu Memory
        (err,) = cudart.cudaFree(inputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(inputs_2[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs_2[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess

        show_cls_result(output)
        show_cls_result(output_2)


def test_build_engine_trtapi_dynamicshape(config):
    onnx_model = os.path.join(config.model_path, "resnet18.onnx")
    IXRT_LOGGER = ixrt.Logger(ixrt.Logger.WARNING)
    builder = ixrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input", Dims([1, 3, 112, 112]), Dims([1, 3, 224, 224]), Dims([1, 3, 448, 448])
    )
    build_config.add_optimization_profile(profile)
    parser = ixrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model)

    if config.precision == "int8":
        build_config.set_flag(ixrt.BuilderFlag.INT8)
    else:
        build_config.set_flag(ixrt.BuilderFlag.FP16)

    # set dynamic
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([-1, 3, -1, -1])

    plan = builder.build_serialized_network(network, build_config)
    if config.precision == "int8":
        engine_file_path = os.path.join(
            config.model_path, "resnet18_trt_int8_dynamicshape.engine"
        )
    else:
        engine_file_path = os.path.join(
            config.model_path, "resnet18_trt_fp16_dynamicshape.engine"
        )
    with open(engine_file_path, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def test_resnet18_trtapi_dynamicshape(config):
    print("====================test_resnet18_trtapi_dynamicshape====================")
    if config.precision == "int8":
        engine_path = os.path.join(
            config.model_path, "resnet18_trt_int8_dynamicshape.engine"
        )
    else:
        engine_path = os.path.join(
            config.model_path, "resnet18_trt_fp16_dynamicshape.engine"
        )
    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.ERROR)
    input_name = "input"
    output_name = "output"
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

        # set_dynamic shape
        input_shape = [1, 3, 224, 224]
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(input_shape))
        # Setup I/O bindings
        inputs, outputs, allocations = setup_io_bindings(engine, context)

        ### infer
        # Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

        data_batch = (
            np.flip(
                cv2.resize(cv2.imread(config.image_file), input_shape[2:]) / 255.0,
                axis=2,
            )
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
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context.execute_async_v2(allocations, stream)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context.execute_v2(allocations)
        assert outputs[0]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output,
            outputs[0]["allocation"],
            outputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        print("input shape:", input_shape)
        show_cls_result(output)
        (err,) = cudart.cudaFree(inputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess

        # set_dynamic shape
        input_shape_2 = [1, 3, 196, 196]
        context.set_binding_shape(input_idx, Dims(input_shape_2))
        # Setup I/O bindings
        inputs_2, outputs_2, allocations_2 = setup_io_bindings(engine, context)

        ### infer
        # Prepare the output data
        output_2 = np.zeros(outputs_2[0]["shape"], outputs_2[0]["dtype"])

        data_batch = (
            np.flip(
                cv2.resize(cv2.imread(config.image_file), input_shape_2[2:]) / 255.0,
                axis=2,
            )
            .astype("float32")
            .transpose(2, 0, 1)
        )
        data_batch = data_batch.reshape(1, *data_batch.shape)
        data_batch = np.ascontiguousarray(data_batch)

        # Process I/O and execute the network
        assert inputs_2[0]["nbytes"] == data_batch.nbytes
        (err,) = cudart.cudaMemcpy(
            inputs_2[0]["allocation"],
            data_batch,
            inputs_2[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context.execute_async_v2(allocations_2, stream)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context.execute_v2(allocations_2)
        assert outputs_2[0]["nbytes"] == output_2.nbytes
        (err,) = cudart.cudaMemcpy(
            output_2,
            outputs_2[0]["allocation"],
            outputs_2[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        print("input shape:", input_shape_2)
        show_cls_result(output_2)
        (err,) = cudart.cudaFree(inputs_2[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs_2[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess


def test_build_engine_trtapi_dynamicshape_multiprofile(config):
    onnx_model = os.path.join(config.model_path, "resnet18.onnx")
    quant_file = os.path.join(config.model_path, "quantized_resnet18.json")
    IXRT_LOGGER = ixrt.Logger(ixrt.Logger.WARNING)
    builder = ixrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input", Dims([1, 3, 112, 112]), Dims([1, 3, 224, 224]), Dims([1, 3, 448, 448])
    )
    profile_2 = builder.create_optimization_profile()
    profile_2.set_shape(
        "input", Dims([1, 3, 112, 112]), Dims([1, 3, 224, 224]), Dims([1, 3, 448, 448])
    )
    # add three profiles
    build_config.add_optimization_profile(profile)
    build_config.add_optimization_profile(profile_2)
    build_config.add_optimization_profile(profile)
    parser = ixrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model)

    if config.precision == "int8":
        build_config.set_flag(ixrt.BuilderFlag.INT8)
    else:
        build_config.set_flag(ixrt.BuilderFlag.FP16)

    # set dynamic
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([-1, 3, -1, -1])

    plan = builder.build_serialized_network(network, build_config)
    if config.precision == "int8":
        engine_file_path = os.path.join(
            config.model_path, "resnet18_trt_int8_dynamicshape.engine"
        )
    else:
        engine_file_path = os.path.join(
            config.model_path, "resnet18_trt_fp16_dynamicshape.engine"
        )
    with open(engine_file_path, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def setup_multicontext_io_bindings(engine, context, binding_cell_size):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []
    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
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
        if engine.binding_is_input(i % binding_cell_size):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations


def test_resnet18_trtapi_dynamicshape_multicontext(config):
    print(
        "====================test_resnet18_trtapi_dynamicshape_multicontext===================="
    )
    if config.precision == "int8":
        engine_path = os.path.join(
            config.model_path, "resnet18_trt_int8_dynamicshape.engine"
        )
    else:
        engine_path = os.path.join(
            config.model_path, "resnet18_trt_fp16_dynamicshape.engine"
        )
    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.ERROR)
    input_name = "input"
    output_name = "output"
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context
        context_2 = engine.create_execution_context()
        assert context_2

        # set_dynamic shape
        input_shape = [1, 3, 224, 224]
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(input_shape))
        # Setup I/O bindings
        inputs, outputs, allocations = setup_io_bindings(engine, context)

        ### infer
        # Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

        data_batch = (
            np.flip(
                cv2.resize(cv2.imread(config.image_file), input_shape[2:]) / 255.0,
                axis=2,
            )
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
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context.execute_async_v2(allocations, stream)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context.execute_v2(allocations)
        assert outputs[0]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output,
            outputs[0]["allocation"],
            outputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        print("input shape:", input_shape)
        show_cls_result(output)
        (err,) = cudart.cudaFree(inputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(outputs[0]["allocation"])
        assert err == cudart.cudaError_t.cudaSuccess

        # Switch optimization profile by set profile index
        binding_cell_size = 2
        profile_idx = 1
        input_shape_2 = [1, 3, 196, 196]
        err, stream_opt_profile = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess
        context_2.set_optimization_profile_async(profile_idx, stream_opt_profile)
        context_2.set_binding_shape(input_idx, Dims(input_shape_2))
        # Setup I/O bindings
        inputs_2, outputs_2, allocations_2 = setup_multicontext_io_bindings(
            engine, context_2, binding_cell_size
        )

        ### infer
        # Prepare the output data
        output_2 = np.zeros(
            outputs_2[0 + profile_idx]["shape"], outputs_2[0 + profile_idx]["dtype"]
        )

        data_batch = (
            np.flip(
                cv2.resize(cv2.imread(config.image_file), input_shape_2[2:]) / 255.0,
                axis=2,
            )
            .astype("float32")
            .transpose(2, 0, 1)
        )
        data_batch = data_batch.reshape(1, *data_batch.shape)
        data_batch = np.ascontiguousarray(data_batch)

        # Process I/O and execute the network
        assert inputs_2[0 + profile_idx]["nbytes"] == data_batch.nbytes
        (err,) = cudart.cudaMemcpy(
            inputs_2[0 + profile_idx]["allocation"],
            data_batch,
            inputs_2[0 + profile_idx]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        if config.use_async:
            err, stream = cudart.cudaStreamCreate()
            assert err == cudart.cudaError_t.cudaSuccess
            context_2.execute_async_v2(allocations_2, stream)
            (err,) = cudart.cudaStreamSynchronize(stream)
            assert err == cudart.cudaError_t.cudaSuccess
            (err,) = cudart.cudaStreamDestroy(stream)
            assert err == cudart.cudaError_t.cudaSuccess
        else:
            context_2.execute_v2(allocations_2)
        assert outputs_2[0 + profile_idx]["nbytes"] == output.nbytes
        (err,) = cudart.cudaMemcpy(
            output_2,
            outputs_2[0 + profile_idx]["allocation"],
            outputs_2[0 + profile_idx]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        print("input shape:", input_shape_2)
        show_cls_result(output_2)
        (err,) = cudart.cudaStreamDestroy(stream_opt_profile)
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(
            inputs_2[0 + profile_idx * binding_cell_size]["allocation"]
        )
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaFree(
            outputs_2[0 + profile_idx * binding_cell_size]["allocation"]
        )
        assert err == cudart.cudaError_t.cudaSuccess


if __name__ == "__main__":
    config = parse_config()

    test_build_engine_trtapi(config)
    test_resnet18_trtapi_ixrt(config)
    if config.multicontext:
        test_resnet18_trtapi_multicontext(config)
    if config.dynamicshape:
        test_build_engine_trtapi_dynamicshape(config)
        test_resnet18_trtapi_dynamicshape(config)
    if config.dynamicshape_multicontext:
        test_build_engine_trtapi_dynamicshape_multiprofile(config)
        test_resnet18_trtapi_dynamicshape_multicontext(config)
