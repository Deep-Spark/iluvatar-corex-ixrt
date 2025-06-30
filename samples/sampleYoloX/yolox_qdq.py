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

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from os.path import basename, dirname, join

import cv2
import numpy as np
import onnx
import torch
from onnxsim import simplify

from ixrt.deploy.api import static_quantize


class AddQdqYoloX:
    def __init__(self, model):
        self.dataloader = None
        self.model = model
        self.sim_model = model[0:-5] + "_sim.onnx"

    def get_calibration_dataset(self, calib_dataset_path):
        files = os.listdir(calib_dataset_path)
        img_files = [join(calib_dataset_path, i) for i in files if i.endswith(".jpg")]
        img_data = [cv2.imread(f) for f in img_files]
        img_nchw = [
            d.transpose((2, 0, 1)).reshape(1, 1, 3, 640, 640).astype(np.float32)
            for d in img_data
        ]
        ndarray = np.concatenate(img_nchw, axis=0)
        self.dataloader = torch.from_numpy(ndarray).cuda()

    def simplify(self):
        onnx_model = onnx.load(self.model)
        model_simp, check = simplify(onnx_model)
        onnx.save(model_simp, self.sim_model)
        print("Simplify onnx Done.")

    def qdq_quantize(self, calib_path, save_quant_onnx_path, save_quant_params_path):
        self.simplify()
        self.get_calibration_dataset(calib_path)
        graph = static_quantize(
            model=self.sim_model,
            calibration_dataloader=self.dataloader,
            save_quant_onnx_path=save_quant_onnx_path,
            save_quant_params_path=save_quant_params_path,
            quant_format="qdq",
        )


def parse_args():
    file = join(
        dirname(__file__),
        "../../data/yolox_m/yolox_m_offical.onnx",
    )
    save_model_path = join(
        dirname(__file__),
        "../../data/yolox_m/yolox_m_qdq_quant.onnx",
    )
    calib_path = join(
        dirname(__file__),
        "../../data/yolox_m",
    )
    save_param_path = join(
        dirname(__file__),
        "../../data/yolox_m/yolox_m_qdq_quant_params.json",
    )
    parser = argparse.ArgumentParser("Qdq quantization")
    parser.add_argument("--model", type=str, default=file)
    parser.add_argument("--calib_path", type=str, default=calib_path)
    parser.add_argument("--save_model_path", type=str, default=save_model_path)
    parser.add_argument("--save_param_path", type=str, default=save_param_path)

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_args()
    quantizer = AddQdqYoloX(model=config.model)
    quantizer.qdq_quantize(
        calib_path=config.calib_path,
        save_quant_onnx_path=config.save_model_path,
        save_quant_params_path=config.save_param_path,
    )
    print("Done, quantized onnx lies on", config.save_model_path)
