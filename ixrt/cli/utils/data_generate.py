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

import os
import sys

import numpy as np
import torch
from ixrt.deploy.api import *
from torch.utils.data import DataLoader, TensorDataset


def generate_data(data_name, data_shape, data_types):
    if data_name not in data_types.keys():
        print("unknown datatypes of ", data_name)
    else:
        data_type = str(data_types[data_name])
        if data_type == "float32":
            np_data = np.ones(data_shape, np.float32)
        elif data_type == "float16":
            np_data = np.ones(data_shape, np.float16)
        elif data_type == "float64":
            np_data = np.ones(data_shape, np.float64)
        elif data_type == "int8":
            np_data = np.ones(data_shape, np.int8)
        elif data_type == "int32":
            np_data = np.ones(data_shape, np.int32)
        elif data_type == "int64":
            np_data = np.ones(data_shape, np.int64)
        else:
            raise ValueError(f"unsupported data type {data_type}")
    return np_data


def generate_buffers(data_bindings, custom_buffers=None):
    data_buffer = {}
    for data_binding in data_bindings:
        _name = data_binding["name"]
        _shape = data_binding["shape"]
        _dtype = data_binding["dtype"]
        if _dtype in (np.float32, np.float16, np.float64):
            buffer = np.random.rand(*_shape).astype(_dtype)
        elif _dtype in (np.int8, np.int16, np.int32, np.int64, np.bool_):
            buffer = np.ones(_shape, _dtype)
        else:
            raise Exception("Not supported data initialization for", _dtype)

        if custom_buffers:
            if _name in custom_buffers:
                buffer = np.fromfile(custom_buffers[_name], _dtype).reshape(_shape)
        data_buffer[_name] = buffer
    return data_buffer


def generate_dummy_calibration_loader(data_shapes, data_types):
    fake_data_list = []
    for data_name, data_shape in data_shapes.items():
        fake_data = generate_data(data_name, data_shape, data_types)
        fake_data_list.append(torch.from_numpy(fake_data))

    calibration_dataset = TensorDataset(*fake_data_list)
    return DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=1,
    )


def generate_quantified_model(
    onnx_model, data_shapes, data_types, save_quant_onnx_path, save_quant_params_path
):
    calibration_dataloader = generate_dummy_calibration_loader(data_shapes, data_types)

    device = 0 if torch.cuda.is_available() else "cpu"
    static_quantize(
        onnx_model,
        calibration_dataloader,
        disable_bias_correction=True,
        analyze=False,
        save_quant_onnx_path=save_quant_onnx_path,
        save_quant_params_path=save_quant_params_path,
        data_preprocess=lambda x: x[0].to(device),
        device=device,
    )
    del calibration_dataloader
