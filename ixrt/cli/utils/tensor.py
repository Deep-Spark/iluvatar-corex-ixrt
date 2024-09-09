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


def _to_double_paddings(single_paddings):
    return [(0, i) for i in single_paddings]


def remove_padding(arr, ixrt_paddings):
    slices = []
    for dim, (before, after) in enumerate(_to_double_paddings(ixrt_paddings)):
        slices.append(slice(before, arr.shape[dim] - after))
    return arr[tuple(slices)]


def add_padding(arr, ixrt_paddings):
    paddings = _to_double_paddings(ixrt_paddings)
    return np.pad(arr, paddings)
