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

from ixrt.deploy.ir import DataType, Graph

IR2IXRT_DTYPE = {
    DataType.UNDEFINED: "float32",
    DataType.FLOAT: "float32",
    DataType.INT8: "int8",
    DataType.UINT8: "uint8",
    DataType.UINT16: "uint16",
    DataType.INT16: "int16",
    DataType.INT32: "int32",
    DataType.INT64: "int64",
    DataType.STRING: "string",
    DataType.BOOL: "bool",
    DataType.FLOAT16: "float16",
    DataType.DOUBLE: "float64",
    DataType.UINT32: "uint32",
    DataType.UINT64: "uint64",
}


def ir_to_ixrt_dtype(dtype: DataType):
    return IR2IXRT_DTYPE.get(dtype, "float32")
