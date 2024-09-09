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

import onnx
import torch
from ixrt.deploy import Pipeline, create_source, create_target, static_quantize
from ixrt.deploy.fusion.level2.yolo_pass import Yolov5Pass


# Fusion
def fuse_model(args):
    dummy_inputs = torch.randn(args.bsz, 3, args.imgsz, args.imgsz)
    refine_pipeline = Pipeline(
        create_source(args.origin_model),
        Yolov5Pass(args.faster_impl),
        create_target(
            args.output_model,
            example_inputs=dummy_inputs,
        ),
    )
    graph = refine_pipeline.run()[0]
    print("  Model Fusion Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--faster_impl", type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()
fuse_model(args)
