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

import inspect
from typing import List, Union

from .base_pass import BasePass, PassSequence, get_pass_registry

__all__ = [
    "get_all_passes",
    "get_level0_passes",
    "get_level1_passes",
    "get_default_passes",
    "create_pass",
    "create_passes",
]


def get_all_passes():
    passes = dict()
    for level in range(3):
        passes.update(get_pass_registry(level))
    return passes


def create_pass(name: str, *args, **kwargs):
    for registry in get_pass_registry().values():
        if registry.containe(name):
            pass_ = registry.get(name)
            if inspect.isclass(pass_):
                return pass_(*args, **kwargs)
            return pass_

    if isinstance(name, BasePass):
        return name

    if not isinstance(name, str):
        return name

    if name == "default":
        return get_default_passes()

    if name.lower() in ["0", "level0"]:
        return get_level0_passes()

    if name.lower() in ["1", "level1"]:
        return get_level1_passes()

    raise RuntimeError(f"Invalid pass, got {name}.")


def create_passes(names: Union[str, List[str]]) -> List[BasePass]:
    if not isinstance(names, (str, list, tuple, BasePass)):
        raise RuntimeError(
            f"The argument `names` must be str, pass or list, but got {names}."
        )

    if isinstance(names, (str, BasePass)):
        return [create_pass(names)]

    passes = []
    for name in names:
        passes.append(create_pass(name))
    return passes


def get_level0_passes():
    passes = list(get_pass_registry(0).get_handlers())
    for i, pass_ in enumerate(passes):
        if inspect.isclass(pass_):
            passes[i] = pass_()
    return PassSequence(*passes)


def get_level1_passes():
    passes = list(get_pass_registry(1).get_handlers())
    for i, pass_ in enumerate(passes):
        if inspect.isclass(pass_):
            passes[i] = pass_()
    return PassSequence(*passes)


def get_default_passes():
    return PassSequence(
        get_level0_passes(),
        get_level1_passes(),
    )
