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

from typing import List, Type, Union

from .base import BaseHook
from .container_hook import ContainerHook


class ListHooks(ContainerHook):
    def __init__(self, *args, **kwargs):
        super(ListHooks, self).__init__(*args, **kwargs)
        self.hooks = []

    def add_hook(self, hook: BaseHook):
        super(ListHooks, self).add_hook(hook)
        self.hooks.append(hook)

    def add_hooks(self, hooks: List[BaseHook]):
        for hook in hooks:
            self.add_hook(hook)

    def remove_hook(self, hook: BaseHook = None, idx: int = None) -> BaseHook:
        super(ListHooks, self).remove_hook()
        if hook is not None:
            self.hooks.remove(hook)
            return hook

        if idx is not None:
            return self.hooks.pop(idx)

        raise ValueError("One of the arguments hook and idx must be given.")

    def get_hooks(self, fn: str = None) -> List[BaseHook]:
        return self.hooks
