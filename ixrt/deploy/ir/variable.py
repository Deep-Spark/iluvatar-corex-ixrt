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

import copy
from typing import Any, List, Optional

from ixrt.deploy.core import PropertyObserver

from .data_type import DataType, VariableType


class VariableOptions(object):
    def __init__(
        self,
        dtype: DataType = None,
        shape: List = None,
        var_type: VariableType = None,
        is_parameter: bool = False,
    ):
        if dtype is None:
            dtype = DataType.UNDEFINED

        self.dtype = dtype
        self.shape = shape
        self.var_type = VariableType.TENSOR if var_type is None else var_type
        self.is_parameter = is_parameter


class Variable(object):
    def __init__(self, name: str, value: Any = None, options: VariableOptions = None):
        self._name = name
        self._value = value

        # This property is used to set multi-outputs.
        self._value_keys = None
        if options is None:
            options = VariableOptions(
                DataType.UNDEFINED, var_type=VariableType.UNDEFINED
            )
        self._options = options
        self._value_observer: Optional[PropertyObserver] = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def value(self):
        return self._value

    @property
    def value_keys(self):
        return self._value_keys

    @value_keys.setter
    def value_keys(self, new_keys):
        self._value_keys = new_keys

    @value.setter
    def value(self, v: Any):
        if self.value_observer is not None:
            self.value_observer.change_before(self._value)

        self._value = v

        if self.value_observer is not None:
            self.value_observer.change_after(v)

    @property
    def value_observer(self):
        return self._value_observer

    def set_value_observer(self, observer: PropertyObserver):
        self._value_observer = observer
        observer.set_watched_variable(self)

    def remove_value_observer(self):
        if self._value_observer is not None:
            self._value_observer.set_watched_variable(None)
        self._value_observer = None

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, v: VariableOptions):
        self._options = v

    @property
    def dtype(self) -> DataType:
        return self.options.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        if isinstance(new_dtype, str):
            self._options.dtype = DataType.from_name(new_dtype)
        elif isinstance(new_dtype, DataType):
            self._options.dtype = new_dtype
        elif isinstance(new_dtype, int):
            self._options.dtype = DataType.from_value(new_dtype)
        else:
            raise ValueError(
                f"new_dtype should be either str or DataType, but got {type(new_dtype)}"
            )

    def set_shape(self, shape: List):
        self.options.shape = shape

    @property
    def var_type(self) -> VariableType:
        return self.options.var_type

    @property
    def shape(self):
        return self.options.shape

    @property
    def is_parameter(self):
        return self.options.is_parameter

    def mark_as_parameter(self):
        self.options.is_parameter = True

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"Variable(name={self.name}, dtype={self.dtype}, shape={self.shape})"


class Placeholder(Variable):
    def __init__(self, name):
        super(Placeholder, self).__init__(name=name)
