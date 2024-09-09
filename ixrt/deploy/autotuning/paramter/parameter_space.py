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

import itertools
import random
from typing import List

from ixrt.deploy.core.registry import Registry, build_from_cfg

from .autotune_parameter import AutoTuneParameter

PARAMETER_SPACE_REGISTRY = Registry("ParameterSpace")


class ParameterSpace(object):
    def __init__(self, param_list: List[AutoTuneParameter]):
        self._param_list = param_list
        self._iter_index = 0
        self.result = None

    @property
    def param_list(self):
        return self._param_list

    def next(self) -> dict:
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()

    def set_result(self, result):
        self.result = result

    def __next__(self):
        if self._iter_index < self.count():
            ret = self.next()
            self._iter_index += 1
            return ret
        else:
            raise StopIteration()

    def __iter__(self):
        self._iter_index = 0
        return self

    def __len__(self):
        return self.count()


@PARAMETER_SPACE_REGISTRY.registe(alias="grid")
class GridParameterSpace(ParameterSpace):
    def __init__(self, *args, **kwargs):
        super(GridParameterSpace, self).__init__(*args, **kwargs)
        self.grid_params = self.generate_grid_params()

    def next(self):
        return self.grid_params[self._iter_index]

    def count(self):
        return len(self.grid_params)

    def generate_grid_params(self):
        grid_params = []
        names = []
        values = []
        for p in self.param_list:
            names.append(p.param_name)
            values.append(p.discrete())

        comb_values = itertools.product(*values)
        for param_values in comb_values:
            grid_params.append({name: p for name, p in zip(names, param_values)})

        return grid_params


@PARAMETER_SPACE_REGISTRY.registe(alias="random")
class RandomParameterSpace(GridParameterSpace):
    def __init__(self, num, *args, **kwargs):
        super(RandomParameterSpace, self).__init__(*args, **kwargs)
        self.grid_params = random.choices(self.grid_params, k=num)


def create_parameter_space(space, **kwargs):
    kwargs["type"] = space
    return build_from_cfg(kwargs, PARAMETER_SPACE_REGISTRY)
