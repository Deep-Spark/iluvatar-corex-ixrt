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

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Type, Union

from .base import BaseHook
from .container_hook import ContainerHook
from .list_hooks import ListHooks

DEFAULT_PRIORITY = 0


class PriorityHooks(ContainerHook):

    """
    Example:
        >>> a = BaseHook()
        >>> b = BaseHook()
        >>> hook = PriorityHooks()
        >>> hook.add_hook(a, priority=1)
        >>> hook.add_hook(b, priority=2)

        Actually calling order where `reversed` is True:
            b.on_epoch_start()
            a.on_epoch_start()
            a.on_epoch_end()
            b.on_epoch_end()

        Actually calling order where `reversed` is False:
            b.on_epoch_start()
            a.on_epoch_start()
            b.on_epoch_end()
            a.on_epoch_end()

    """

    def __init__(self, reversed=True, *args, **kwargs):
        super(PriorityHooks, self).__init__(*args, **kwargs)
        self.reversed = reversed
        self.hooks: Dict[int, ListHooks] = defaultdict(ListHooks)

        self._flatten_hooks_list = None

    def add_hook(self, hook: BaseHook, priority: Optional[int] = None):
        if priority is None:
            priority = DEFAULT_PRIORITY
        super(PriorityHooks, self).add_hook(hook)
        self.hooks[priority].add_hook(hook)

    def add_hooks(self, hooks: List[BaseHook], priorities: List[int] = None):
        if priorities is None:
            priorities = [None] * len(hooks)

        for hook, priority in zip(hooks, priorities):
            self.add_hook(hook, priority)

    def remove_hook(self, hook) -> BaseHook:
        super(PriorityHooks, self).remove_hook()
        for priority, hooks in self.hooks.items():
            if hook in hooks.hooks:
                return hooks.remove_hook(hook=hook)

    def get_hooks(self, fn: str = "") -> List[BaseHook]:
        if self._flatten_hooks_list is None or not self.is_frozen:
            self._flatten_hooks_list = self.flatten_hooks()

        if self.reversed and fn.endswith("_end"):
            return list(reversed(self._flatten_hooks_list))

        return self._flatten_hooks_list

    def flatten_hooks(self):
        priotity_list = list(self.hooks.keys())
        priotity_list.sort(reverse=True)
        flatten_hooks = []
        for priotity in priotity_list:
            flatten_hooks.append(self.hooks[priotity])

        return flatten_hooks
