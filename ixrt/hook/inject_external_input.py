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

from functools import partial

import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np
import ixrt

from .utils import add_padding, quant

__all__ = ["inject_external_input"]


def chw2hwc(arr):
    arr = arr.transpose(0, *range(2, arr.ndim), 1)
    return np.ascontiguousarray(arr)


def inject_external_input(info, external_input):
    """Inject external data with linear format to IxRT runtime input

    :param info:
    :param external_input: dict[str, str], key is edge name, value is path
    :return:
    """
    for i in range(info.nb_inputs):
        iname = info.input_names[i]
        if iname in external_input:
            arr = np.load(external_input[iname])
            tensori = info.input_tensors[i]

            # add padding
            arr = add_padding(arr, tensori.paddings)
            if tensori.shape != arr.shape:
                print(
                    f"Injected tensor has shape {arr.shape}, not match ixrt tensor shape {tensori.shape}"
                )
                raise

            # channel switch
            if tensori.format == ixrt.TensorFormat.HWC:
                arr = chw2hwc(arr)
            # type conversion
            if tensori.dtype == ixrt.DataType.HALF:
                arr = arr.astype(np.float16)
            elif tensori.dtype == ixrt.DataType.INT8:
                arr = quant(arr, tensori.scale)
            else:
                if not ixrt.nptype(tensori.dtype) == arr.dtype:
                    raise Exception(
                        f"The external input tensor {external_input[iname]} has different dtype that IxRT required, given: {arr.dtype}, require: {ixrt.nptype(tensori.dtype)}"
                    )

            (err,) = cudart.cudaMemcpy(
                tensori.data,
                arr,
                arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            assert err == cudart.cudaError_t.cudaSuccess
