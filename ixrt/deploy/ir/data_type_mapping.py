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

from typing import Union

import numpy as np
import onnx
import torch

from .data_type import DataType
from .variable import VariableType

_TORCH2IR = {
    torch.float16: DataType.FLOAT16,
    torch.half: DataType.FLOAT16,
    torch.float: DataType.FLOAT,
    torch.float32: DataType.FLOAT,
    torch.float64: DataType.DOUBLE,
    torch.double: DataType.DOUBLE,
    torch.uint8: DataType.UINT8,
    torch.int8: DataType.INT8,
    torch.int16: DataType.INT16,
    torch.int32: DataType.INT32,
    torch.int64: DataType.INT64,
    torch.qint8: DataType.INT8,
    torch.quint8: DataType.UINT8,
    torch.qint32: DataType.INT32,
    torch.complex32: DataType.COMPLEX32,
    torch.complex64: DataType.COMPLEX64,
    torch.complex128: DataType.COMPLEX128,
    torch.bool: DataType.BOOL,
    torch.bfloat16: DataType.BFLOAT16,
}


NUMPY2IR = {
    "float16": DataType.FLOAT16,
    "float": DataType.FLOAT,
    "float32": DataType.FLOAT,
    "float64": DataType.DOUBLE,
    "double": DataType.DOUBLE,
    "uint8": DataType.UINT8,
    "int": DataType.INT32,
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "uint": DataType.UINT32,
    "uint32": DataType.UINT32,
    "uint64": DataType.UINT64,
    "bool": DataType.BOOL,
}


__all__ = [
    "onnx_to_ir_dtype",
    "onnx_to_ir_var_type",
    "torch_to_ir_dtype",
    "py_to_onnx_dtype",
    "ir_to_torch_dtype",
]


def onnx_to_ir_dtype(onnx_data_type):
    return DataType.from_value(onnx_data_type)


def onnx_to_ir_var_type(type: onnx.TypeProto) -> VariableType:
    onnx_type2_ir_type = [
        ("tensor", VariableType.TENSOR),
        ("sequence", VariableType.LIST),
        ("map", VariableType.MAP),
        ("optional", VariableType.OPTIONAL),
        ("sparse_tensor", VariableType.SPARSE),
    ]

    for from_type, to_type in onnx_type2_ir_type:
        mate = f"{from_type}_type"
        if getattr(type, mate).ByteSize() != 0:
            return VariableType(to_type, mate)

    raise RuntimeError(f"The type `{type}` is not defined.")


def py_to_onnx_dtype(dtype):
    dtype_mapping = {
        bool: DataType.BOOL,
        float: DataType.FLOAT,
        int: DataType.INT32,
        str: DataType.STRING,
    }
    return dtype_mapping.get(dtype, DataType.UNDEFINED)


def torch_to_ir_dtype(dtype):
    if _TORCH2IR is None:
        raise ImportError("The torch package is not installed.")

    return _TORCH2IR[dtype]


def onnx_to_torch_dtype(dtype):
    if not isinstance(dtype, DataType):
        dtype = DataType.from_value(dtype)

    for torch_type, ir_type in _TORCH2IR.items():
        if ir_type == dtype:
            return torch_type

    raise RuntimeError(f"Not found data type {dtype}.")


def ir_to_torch_dtype(dtype, is_quant_type=False):
    if isinstance(dtype, DataType):
        dtype = dtype.name

    for _type, _ir_value in _TORCH2IR.items():
        name: str = DataType.from_value(_ir_value).name
        if name.lower() == dtype.lower():
            if is_quant_type:
                if str(_type).startswith("torch.q"):
                    return _type
            else:
                return _type

    raise RuntimeError(f"Cannot convert `{dtype}` to torch dtype.")


def ir_to_onnx_dtype(dtype):
    return DataType.from_value(dtype).value


def ir_to_numpy_dtype(dtype: DataType):
    for np_dtype, ir_dtype in NUMPY2IR.items():
        if ir_dtype == dtype:
            return np.dtype(np_dtype)
    raise RuntimeError(f"Not support data type, got {dtype.name}.")


def get_dtype_from_tensor(tensor: Union[torch.Tensor, np.ndarray]):
    if not torch.is_tensor(tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor.copy())
        else:
            tensor = torch.tensor(tensor)
    return torch_to_ir_dtype(tensor.dtype)
