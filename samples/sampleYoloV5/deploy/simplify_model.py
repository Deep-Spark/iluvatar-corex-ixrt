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
from onnxsim import simplify


# Simplify
def simplify_model(args):
    onnx_model = onnx.load(args.origin_model)
    model_simp, check = simplify(onnx_model)
    model_simp = onnx.shape_inference.infer_shapes(model_simp)
    onnx.save(model_simp, args.output_model)
    print("  Simplify onnx Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args


args = parse_args()
simplify_model(args)
