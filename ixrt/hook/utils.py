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

from ixrt import volume


# quant
def quant(arr, scale):
    if not scale:
        raise Exception("Tried to quant tensor but got empty scale")
    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    arr = arr / scale
    arr = arr.round()
    arr = np.clip(arr, -128, 127)
    return arr.astype(np.int8)


def dequant(arr, scale):
    if not scale:
        raise Exception("Tried to dequant tensor but got empty scale")
    ndims = len(arr.shape)
    scale = np.array(scale).reshape(-1, *[1] * (ndims - 1))
    return (arr * scale).astype(np.float32)


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

def apply_format(shape, format):
    if format == ixrt.TensorFormat.HWC and len(shape)>2:
        return [shape[0], *shape[2:], shape[1]]
    return shape
def convert_shape_tensor(tensor):
    import ctypes
    pointer = None
    if tensor.dtype == ixrt.DataType.INT64:
        pointer = ctypes.c_int64 * volume(tensor.shape)
    elif tensor.dtype == ixrt.DataType.INT32:
        pointer = ctypes.c_int64 * volume(tensor.shape)
    elif tensor.dtype == ixrt.DataType.FLOAT:
        pointer = ctypes.c_float * volume(tensor.shape)
    elif tensor.dtype == ixrt.DataType.BOOL:
        pointer = ctypes.c_bool * volume(tensor.shape)
    else:
        raise Exception("Unsupported tensor dtype for", tensor.dtype)

    c_array = pointer.from_address(tensor.data)
    return np.frombuffer(c_array, dtype=ixrt.nptype(tensor.dtype))

def convert_execution_tensor(tensor, ort_style=True):
    result = np.zeros(apply_format(tensor.shape, tensor.format), ixrt.nptype(tensor.dtype))
    (err,) = cudart.cudaMemcpy(
        result,
        tensor.data,
        result.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    if ort_style:
        # process tensor such that its datatype can be the same as fp32 inference of ort
        # currently, we only have two rules:
        # int8 --> float32
        # float16 --> float32
        # if result.dtype == np.int8:
        #     result = dequant(result, tensor.scale)
        if result.dtype == np.float16:
            result = result.astype(np.float32)

        # process tensor data format to be linear, which is consistent with ort
        if tensor.format == ixrt.TensorFormat.HWC and len(tensor.shape) >= 3:
            result = result.transpose(
                0, result.ndim - 1, *range(1, result.ndim - 1)
            )
        # remove padding
        result = remove_padding(result, tensor.paddings)
    return result
def copy_ixrt_tensor_as_np(tensor, ort_style=True):
    if tensor.is_shape_tensor:
        return convert_shape_tensor(tensor)
    else:
        return convert_execution_tensor(tensor, ort_style)



def copy_ixrt_io_tensors_as_np(info, ort_style=True):
    result = dict(input=[], output=[])
    for name, tensor in zip(info.input_names, info.input_tensors):
        result["input"].append((name, copy_ixrt_tensor_as_np(tensor, ort_style)))
    for name, tensor in zip(info.output_names, info.output_tensors):
        result["output"].append((name, copy_ixrt_tensor_as_np(tensor, ort_style)))
    return result
