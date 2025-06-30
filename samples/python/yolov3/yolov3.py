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
import glob
import json
import os
import sys
import time

import cuda.cudart as cudart
import numpy as np
from coco_labels import coco80_to_coco91_class, labels
from common import post_process_withoutNMS, precess_batch_input
from tqdm import tqdm
from tqdm.contrib import tzip

import ixrt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def show_det_result(box_datas, box_nums, threshold):
    det_box_data = box_datas[0, : box_nums[0], :]
    class_map = coco80_to_coco91_class()
    print(
        "------------------------------Python inference result------------------------------"
    )
    for i in range(box_nums[0]):
        if det_box_data[i][5] > threshold:
            det_label = labels[int(det_box_data[i][4]) - 1]
            print(
                f"detect {det_label}:   {det_box_data[i][5]} in  {det_box_data[i][:5]}"
            )


def run_with_engine(config):
    if config.precision == "int8":
        engine_path = os.path.join(config.model_path, "yolov3_trt_int8.engine")
    else:
        engine_path = os.path.join(config.model_path, "yolov3_trt_fp16.engine")
    datatype = ixrt.DataType.FLOAT
    host_mem = ixrt.IHostMemory
    logger = ixrt.Logger(ixrt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        runtime = ixrt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Setup I/O bindings
    inputs = []
    outputs = {}
    allocations = []

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(ixrt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        err, allocation = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        binding = {
            "dtype": np.dtype(ixrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
            "nbytes": size,
        }
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs[name] = binding

    input_dict = {"images": ([config.bsz, 3, 416, 416], "float32")}
    box_datas = np.zeros(
        outputs["nms_output0"]["shape"], outputs["nms_output0"]["dtype"]
    )
    box_nums = np.zeros(
        outputs["int_output1"]["shape"], outputs["int_output1"]["dtype"]
    )

    files = [os.path.join(config.model_path, "dog_416.bmp")]
    input_io_batch, all_img_shape = precess_batch_input(files, config, 1)
    data_batch = input_io_batch[0]

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
        (err,) = cudart.cudaStreamSynchronize(stream)
        assert err == cudart.cudaError_t.cudaSuccess
        (err,) = cudart.cudaStreamDestroy(stream)
        assert err == cudart.cudaError_t.cudaSuccess
    else:
        context.execute_v2(allocations)
    assert outputs["nms_output0"]["nbytes"] == box_datas.nbytes
    (err,) = cudart.cudaMemcpy(
        box_datas,
        outputs["nms_output0"]["allocation"],
        outputs["nms_output0"]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    assert outputs["int_output1"]["nbytes"] == box_nums.nbytes
    (err,) = cudart.cudaMemcpy(
        box_nums,
        outputs["int_output1"]["allocation"],
        outputs["int_output1"]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    (err,) = cudart.cudaFree(inputs[0]["allocation"])
    assert err == cudart.cudaError_t.cudaSuccess
    (err,) = cudart.cudaFree(outputs["nms_output0"]["allocation"])
    assert err == cudart.cudaError_t.cudaSuccess
    (err,) = cudart.cudaFree(outputs["int_output1"]["allocation"])
    assert err == cudart.cudaError_t.cudaSuccess
    print("box_datas", box_datas.shape)
    print("box_nums", box_nums)
    show_det_result(box_datas, box_nums, config.threshold)


def build_engine_trtapi(config):
    onnx_model = os.path.join(config.model_path, "quantized_yolov3_decoder_nms.onnx")
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
        engine_file_path = os.path.join(config.model_path, "yolov3_trt_int8.engine")
    else:
        engine_file_path = os.path.join(config.model_path, "yolov3_trt_fp16.engine")
    with open(engine_file_path, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(ROOT, "data/resnet18"),
        help="model forder path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "int8"],
        default="int8",
        help="The precision of datatype",
    )

    parser.add_argument("--bsz", type=int, default=1, help="test batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="test threshold")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=416,
        help="inference size h,w",
    )
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--use_letterbox", action="store_true")
    parser.add_argument(
        "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4"
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_config()

    build_engine_trtapi(config)
    run_with_engine(config)
