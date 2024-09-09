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
from functools import partial

from .inject_external_input import inject_external_input
from .print_info import print_info


@dataclass
class HookRegistry:
    func: object
    args: dict


available_hooks = {
    "print_info": HookRegistry(print_info, {}),
    "inject_external_input": HookRegistry(inject_external_input, {}),
}


def create_hook(name, **kwargs):
    if name not in available_hooks:
        raise KeyError(f"No hook creator called {name}!")
    return partial(
        available_hooks[name].func, **kwargs if kwargs else available_hooks[name].args
    )
