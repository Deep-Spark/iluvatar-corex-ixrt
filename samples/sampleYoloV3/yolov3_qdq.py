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
import sys
from os.path import basename, dirname, join

import torch
from ixrt.deploy.api import static_quantize


class AddQdqYoloV3:
    def __init__(self, model):
        self.dataloader = torch.randn(2, 1, 3, 416, 416).cuda()
        self.model = model

    def qdq_quantize(self, save_quant_onnx_path, save_quant_params_path):
        graph = static_quantize(
            model=self.model,
            calibration_dataloader=self.dataloader,
            save_quant_onnx_path=save_quant_onnx_path,
            save_quant_params_path=save_quant_params_path,
            quant_format="qdq",
        )


def parse_args():
    file = join(
        dirname(__file__),
        "../../data/yolov3/yolov3_without_decoder.onnx",
    )
    save_model_path = join(
        dirname(__file__),
        "../../data/yolov3/yolov3_without_decoder_qdq.onnx",
    )
    save_param_path = join(
        dirname(__file__),
        "../../data/yolov3/yolov3_quant_params.json",
    )
    parser = argparse.ArgumentParser("Qdq quantization")
    parser.add_argument("--model", type=str, default=file)
    parser.add_argument("--save_model_path", type=str, default=save_model_path)
    parser.add_argument("--save_param_path", type=str, default=save_param_path)

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_args()
    quantizer = AddQdqYoloV3(model=config.model)
    quantizer.qdq_quantize(
        save_quant_onnx_path=config.save_model_path,
        save_quant_params_path=config.save_param_path,
    )
    print("Done, quantized onnx lies on", config.save_model_path)
