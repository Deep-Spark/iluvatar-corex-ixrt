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
import weakref
from abc import abstractmethod

from ixrt.deploy.core import LambdaPropertyObserver, Registry

from ...ir import DataType
from ..quant_paramter import QuantParameter

QUANT_OBSERVERS = Registry("QuantizationObserver")


class QuantVariableObserver(LambdaPropertyObserver):
    def __init__(self, quant_policy=None):
        super(QuantVariableObserver, self).__init__(change_after=self.__call__)
        if quant_policy is not None and quant_policy.qtype != DataType.INT8:
            raise RuntimeError(
                f"Not support the type `{quant_policy.qtype.name}` of quantization."
            )

        self.quant_policy = quant_policy
        self._tensor_t = None

    def __call__(self, *args, **kwargs):
        return self.on_watch(*args, **kwargs)

    @abstractmethod
    def on_watch(self, new_value):
        raise NotImplemented()

    @abstractmethod
    def get_quant_parameters(self) -> QuantParameter:
        pass

    def copy(self):
        return copy.deepcopy(self)

    def mark_as_activation(self):
        self._tensor_t = "activation"

    def is_activation(self):
        return self._tensor_t == "activation"

    def mark_as_weight(self):
        self._tensor_t = "weight"

    def is_weight(self):
        return self._tensor_t == "weight"

    def mark_as_bias(self):
        self._tensor_t = "bias"

    def is_bias(self):
        return self._tensor_t == "bias"


@QUANT_OBSERVERS.registe(alias="identity")
class IdentityObserverType(QuantVariableObserver):
    def on_watch(self, new_value):
        pass

    def get_quant_parameters(self):
        pass


IDENTITY_OBSERVER = IdentityObserverType()


def create_observer(name: str, *args, **kwargs):
    if name.lower() in ["identity"]:
        return IDENTITY_OBSERVER

    return QUANT_OBSERVERS.get(name)(*args, **kwargs)
