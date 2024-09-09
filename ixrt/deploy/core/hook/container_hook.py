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

from abc import abstractmethod
from collections import defaultdict
from typing import Callable, List, Type, Union

from .base import BaseHook
from .wrapper_hook import WrapperHook


class ContainerHook(WrapperHook):
    def __init__(self, *args, **kwargs):
        super(ContainerHook, self).__init__(*args, **kwargs)
        self.hooks = None
        self.is_frozen = False
        self._finished_replace_hook_event = False
        self._name_to_hooks = defaultdict(list)

    def wrap_fn(self, fn) -> Callable:
        def wrapper(*args, **kwargs):
            hooks = self.get_hooks(fn)
            for hook in hooks:
                _hook_event = getattr(hook, fn)
                _hook_event(*args, **kwargs)

        return wrapper

    @abstractmethod
    def get_hooks(self, fn: str = None) -> List[BaseHook]:
        raise NotImplementedError()

    def add_hook(self, hook: BaseHook, *args, **kwargs):
        if not self._finished_replace_hook_event:
            self.replace_hook_events(hook)
        self._finished_replace_hook_event = True
        self._name_to_hooks[type(hook).__name__].append(hook)

    def remove_hook(self, *args, **kwargs) -> BaseHook:
        pass

    @abstractmethod
    def add_hooks(self, hooks: List[BaseHook]):
        raise NotImplementedError()

    def find_hook(self, hook_name: str) -> List[BaseHook]:
        return self._name_to_hooks[hook_name]

    def exist(self, hook: Union[BaseHook, Type[BaseHook], str]) -> bool:
        for _hook in self.get_hooks():
            if _hook == hook:
                return True

            if isinstance(_hook, ContainerHook):
                if _hook.exist(hook):
                    return True
            elif self._is_same_hook(_hook, hook):
                return True
        return False

    def frozen_hooks(self):
        self.is_frozen = True

    def _is_same_hook(
        self, hook: BaseHook, target_hook: Union[str, Type[BaseHook], BaseHook]
    ):
        if isinstance(target_hook, str) and type(hook).__name__ == target_hook:
            return True

        if type(hook) == target_hook:
            return True

        if hook == target_hook:
            return True

        return False
