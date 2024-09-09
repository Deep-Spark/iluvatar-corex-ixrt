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

import numpy as np


def quant(arr, scale):
    if not scale:
        return None

    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    arr = arr / scale
    arr = arr.round()
    arr = np.clip(arr, -128, 127)
    return arr.astype(np.int8)


def dequant(arr, scale):
    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    return (arr * scale).astype(np.float32)
