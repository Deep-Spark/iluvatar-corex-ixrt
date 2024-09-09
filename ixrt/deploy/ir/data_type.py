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

# Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

from enum import IntEnum
from typing import Any, Union


class DataType(IntEnum):
    UNDEFINED = 0
    FLOAT = 1  # float
    UINT8 = 2  # uint8_t
    INT8 = 3  # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5  # int16_t
    INT32 = 6  # int32_t
    INT64 = 7  # int64_t
    STRING = 8  # string
    BOOL = 9  # bool

    # IEEE754 half-precision floating-point format (16 bits wide).
    # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10

    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14  # complex with float32 real and imaginary components
    COMPLEX128 = 15  # complex with float64 real and imaginary components

    # Non-IEEE floating-point format based on IEEE754 single-precision
    # floating-point number truncated to 16 bits.
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16

    FP8 = 17

    # Additional
    COMPLEX32 = 18

    @classmethod
    def from_value(cls, dtype):
        for v in cls.__dict__.values():
            if isinstance(v, DataType) and v.value == dtype:
                return v
        raise RuntimeError(f"The undefined type {dtype}.")

    @classmethod
    def from_name(cls, name: str):
        for dtype_name in cls.__dict__.keys():
            if dtype_name.lower() == name.lower():
                return cls.__dict__[dtype_name]
        raise RuntimeError(f"The undefined type {name}.")


class VariableType(object):
    UNDEFINED = 0
    SCALAR = 1
    TENSOR = 2
    LIST = 3
    MAP = 4
    OPTIONAL = 5
    SPARSE = 6
    MULTI_OUTPUTS = 7

    def __init__(self, type, meta: Any = None):
        self.type = type
        self.mate = meta

    def __eq__(self, other):
        if isinstance(other, VariableType):
            return self.type == other.type

        return self.type == other

    def __repr__(self):
        return f"VariableType(type={self.type}, meta={self.mate})"


_bit_width_mapping = {
    8: [DataType.UINT8.name, DataType.INT8.name, DataType.FP8.name],
    16: [
        DataType.UINT16.name,
        DataType.INT16.name,
        DataType.FLOAT16.name,
        DataType.BFLOAT16.name,
    ],
    32: [DataType.INT32.name, DataType.UINT32.name, DataType.FLOAT.name],
    64: [DataType.DOUBLE.name, DataType.INT64.name, DataType.UINT64.name],
}


def get_type_bit_width(dtype: Union[str, DataType]):
    if isinstance(dtype, DataType):
        dtype = dtype.name.lower()
    dtype = dtype.upper()
    for bit_width, _types in _bit_width_mapping.items():
        if dtype in _types:
            return bit_width
    raise RuntimeError(f"Undefined data type `{dtype}` in _bit_width_mapping.")
