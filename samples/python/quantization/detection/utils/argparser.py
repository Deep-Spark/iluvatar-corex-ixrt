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

from argparse import ArgumentParser


def create_argparser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--observer", type=str, default="hist_percentile")
    parser.add_argument("--fp32_acc", action="store_true")
    parser.add_argument("--use_ixrt", action="store_true")
    # parser.add_argument("--quant_params", type=str, default=None)
    parser.add_argument("--disable_bias_correction", action="store_true")
    return parser
