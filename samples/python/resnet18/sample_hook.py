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
import re
import time

import cuda.cudart as cudart
import cv2
import numpy as np
import ixrt
import torch
from ixrt import Dims
from ixrt.hook.utils import copy_ixrt_io_tensors_as_np
from ixrt.utils import topk
from utils.imagenet_labels import labels as imagenet_labels

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def callback(info):
    print("Layer Type: ", info.layer_type)
    print("OpType: ", info.op_type)
    print("Op: ", info.op_name)
    print("Input names: ", info.input_names)
    print("Output names: ", info.output_names)
    print("Nb inputs: ", info.nb_inputs)
    print("Nb outputs: ", info.nb_outputs)
    print("Input tensors: ", info.input_tensors)
    print("Input tensors[0].dtype: ", info.input_tensors[0].dtype)
    print("Input tensors[0].shape: ", info.input_tensors[0].shape)
    print("Input tensors[0].data: ", info.input_tensors[0].data)
    print("Input tensors[0].scale: ", info.input_tensors[0].scale)
    print("Input tensors[0].itemsize: ", info.input_tensors[0].itemsize)
    io_tensors = copy_ixrt_io_tensors_as_np(info, ort_style=True)
    print(io_tensors)


def callback1(info):
    print("Hook:")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_path",
        type=str,
        default="/tmp/mobilenet_v2.engine",
        help="engine path",
    )
    config = parser.parse_args()
    config.image_file = os.path.join(ROOT, "data", "resnet18", "kitten_224.bmp")
    return config


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
        print(f"Top {i+1}:   {val},{idx}  {labels[idx]}")


def load_dynamic_engine(config):
    print("====================test dynamic engine====================")
    engine_path = config.engine_path

    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.VERBOSE)
    input_name = "input"
    output_name = "output"
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime = ixrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context
        context.register_hook(
            "simple_print", callback, ixrt.ExecutionHookFlag.POSTRUN
        )
        context.register_hook(
            "prerun_hook", callback1, ixrt.ExecutionHookFlag.PRERUN
        )
        # context.deregister_hook("simple_print")
        # context.deregister_hook("prerun_hook")
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


if __name__ == "__main__":
    load_dynamic_engine(parse_config())
