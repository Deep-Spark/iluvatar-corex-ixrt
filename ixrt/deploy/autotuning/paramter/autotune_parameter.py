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

from typing import Callable

import numpy as np


class AutoTuneParameter(object):
    def __init__(self, param_name: str):
        self.param_name = param_name

    def discrete(self):
        raise NotImplementedError()


class DiscreteParameter(AutoTuneParameter):
    def __init__(self, param_name: str, values: list):
        super(DiscreteParameter, self).__init__(param_name)
        self.values = values

    def discrete(self):
        return self.values


class ContinuousParameter(AutoTuneParameter):
    def __init__(
        self,
        param_name: str,
        start: float,
        end: float,
        num: int = 5,
        sampler: Callable = None,
    ):
        super(ContinuousParameter, self).__init__(param_name)
        self.start = start
        self.end = end
        self.num = num
        self.sampler = sampler

    def discrete(self):
        if self.sampler is None:
            return np.linspace(self.start, self.end, self.num, dtype="float32")
        else:
            return self.sampler(self.start, self.end, self.num)
