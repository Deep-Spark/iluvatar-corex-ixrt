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

import cv2
import numpy as np
import torch

from ixrt.utils import topk

from .imagenet_labels import labels

# ../../common.py
parent_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir
)
sys.path.insert(1, parent_dir)
import common


def show_cls_result(result, k=5):
    data = result[0][1]

    vals, idxs = topk(data, k, axis=1)
    # n_samples = len(idxs)
    n_samples = 1
    for i in range(n_samples):
        idx0 = idxs[i]
        val0 = vals[i]
        print("---Python inference result---")
        for i, (val, idx) in enumerate(zip(val0, idx0)):
            print(f"Top {i+1}:   {val}  {labels[idx]}")


def check_cls_result(result, answer, k=5):
    data = result[0][1]
    from imagenet_labels import labels

    vals, idxs = topk(data, k, axis=1)
    idx0 = idxs[0]
    val0 = vals[0]
    answer_ids = []
    answer_names = []
    for _, (val, idx) in enumerate(zip(val0, idx0)):
        answer_ids.append(val)
        answer_names.append(labels[idx])
    if answer not in answer_names:
        return False
    else:
        return True
