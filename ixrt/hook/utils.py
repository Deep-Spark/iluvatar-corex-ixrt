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

from collections import OrderedDict

import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np
import ixrt

__all__ = [
    "quant",
    "dequant",
    "remove_padding",
    "add_padding",
    "copy_ixrt_tensor_as_np",
    "copy_ixrt_io_tensors_as_np",
]

# quant
def quant(arr, scale):
    if not scale:
        raise Exception("Tried to quant tensor but got empty scale")
    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    arr = arr * scale
    arr = arr.round()
    arr = np.clip(arr, -128, 127)
    return arr.astype(np.int8)


def dequant(arr, scale):
    if not scale:
        raise Exception("Tried to dequant tensor but got empty scale")
    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    return (arr / scale).astype(np.float32)


# padding
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


def copy_ixrt_tensor_as_np(tensor, ort_style=True):
    np_array = np.zeros(tensor.shape, ixrt.nptype(tensor.dtype))
    (err,) = cudart.cudaMemcpy(
        np_array,
        tensor.data,
        np_array.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    if ort_style:
        # process tensor such that its datatype can be the same as fp32 inference of ort
        # currently, we only have two rules:
        # int8 --> float32
        # float16 --> float32
        if np_array.dtype == np.int8:
            np_array = dequant(np_array, tensor.scale)
        elif np_array.dtype == np.float16:
            np_array = np_array.astype(np.float32)
        # remove padding
        np_array = remove_padding(np_array, tensor.paddings)
        # process tensor data format to be linear, which is consistent with ort
        if tensor.format == ixrt.TensorFormat.HWC:
            assert (len(tensor.shape) >= 3 or len(tensor.shape) == 1)
            if len(tensor.shape) == 1:
                np_array = np_array
            else:
                np_array = np_array.transpose(
                0, np_array.ndim - 1, *range(1, np_array.ndim - 1))
    return np_array


def copy_ixrt_io_tensors_as_np(info, ort_style=True):
    result = dict(input=[], output=[])
    for name, tensor in zip(info.input_names, info.input_tensors):
        result["input"].append((name, copy_ixrt_tensor_as_np(tensor, ort_style)))
    for name, tensor in zip(info.output_names, info.output_tensors):
        result["output"].append((name, copy_ixrt_tensor_as_np(tensor, ort_style)))
    return result
