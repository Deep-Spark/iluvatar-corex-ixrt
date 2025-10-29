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
import warnings

from ixrt._C import *

from .version import __version__
import sys


# Provides Python's `with` syntax
def common_enter(this):
    warnings.warn(
        "Context managers for IxRT types are deprecated. "
        "Memory will be freed automatically when the reference count reaches 0.",
        DeprecationWarning,
    )
    return this


def common_exit(this, exc_type, exc_value, traceback):
    """
    Context managers are deprecated and have no effect. Objects are automatically freed when
    the reference count reaches 0.
    """
    pass


INetworkDefinition.__enter__ = common_enter
INetworkDefinition.__exit__ = common_exit

IBuilderConfig.__enter__ = common_enter
IBuilderConfig.__exit__ = common_exit

ICudaEngine.__enter__ = common_enter
ICudaEngine.__exit__ = common_exit

IExecutionContext.__enter__ = common_enter
IExecutionContext.__exit__ = common_exit

Runtime.__enter__ = common_enter
Runtime.__exit__ = common_exit

IHostMemory.__enter__ = common_enter
IHostMemory.__exit__ = common_exit

# Add logger severity into the default implementation to preserve backwards compatibility.
Logger.Severity = ILogger.Severity

for attr, value in ILogger.Severity.__members__.items():
    setattr(Logger, attr, value)


def volume(iterable):
    vol = 1
    for elem in iterable:
        vol *= elem
    return vol


# Converts a IxRT datatype to the equivalent numpy type.
def nptype(ixrt_type):
    """
    Returns the numpy-equivalent of a IxRT :class:`DataType` .

    :arg trt_type: The IxRT data type to convert.

    :returns: The equivalent numpy type.
    """
    import numpy as np

    mapping = {
        float16: np.float16,
        float32: np.float32,
        int8: np.int8,
        uint8: np.uint8,
        int32: np.int32,
        int64: np.int64,
        bool: np.bool_,
        bfloat16: np.uint16,
    }
    if ixrt_type in mapping:
        return mapping[ixrt_type]
    raise TypeError("Could not resolve IxRT datatype to an equivalent numpy datatype.")


# Add a numpy-like itemsize property to the datatype.
def _itemsize(trt_type):
    """
    Returns the size in bytes of this :class:`DataType` .

    :arg trt_type: The IxRT data type.

    :returns: The size of the type.
    """
    mapping = {
        float32: 4,
        float16: 2,
        bfloat16: 2,
        int8: 1,
        int32: 4,
        int64: 8,
        bool: 1,
        uint8: 1,
        fp8: 1,
    }
    if trt_type in mapping:
        return mapping[trt_type]


DataType.itemsize = property(lambda this: _itemsize(this))
