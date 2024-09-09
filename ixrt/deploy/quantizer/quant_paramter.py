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

from dataclasses import dataclass
from typing import List, Union

from ..core import BaseDataClass

Vector = List[float]

INF = float("inf")


@dataclass()
class QuantParameter(BaseDataClass):
    per_channel: bool

    scale: Union[float, Vector]
    zero_point: float = 0

    tensor_max: Union[float, Vector] = None
    tensor_min: Union[float, Vector] = None

    qtype_max: Union[float, int] = None
    qtype_min: Union[float, int] = None

    qtype: str = "int8"

    quant_dim: int = -1
    symmetrical: bool = True

    def __post_init__(self):
        self._init_tensor_range()

    def _init_tensor_range(self):
        if self.tensor_min is None or self.tensor_max is None:
            if self.per_channel:
                self.tensor_min = len(self.scale) * [-INF]
                self.tensor_max = len(self.scale) * [INF]
            else:
                self.tensor_min = -INF
                self.tensor_max = INF

        if self.qtype == "int8":
            if self.qtype_min is None:
                self.qtype_min = -127
            if self.qtype_max is None:
                self.qtype_max = 127


def make_param_from_policy(policy):
    # TODO
    pass
