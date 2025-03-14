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
from abc import abstractmethod

from ixrt.deploy.utils.object import get_obj_funcs

from .base import BaseHook
from .replaceable_event_hook import ReplaceableEventHook


class WrapperHook(ReplaceableEventHook):
    def __init__(self):
        super(WrapperHook, self).__init__()

    def replace_hook_events(self, hook):
        def issubclass_of(cls, base):
            for i in cls.__mro__:
                if base in str(i):
                    return True
            return False
        if inspect.isclass(type(hook)) and issubclass_of(type(hook), "BaseHook"):
            funcs = get_obj_funcs(hook).keys()
        elif isinstance(hook, list):
            funcs = hook
        elif isinstance(hook, dict):
            funcs = hook.keys()
        else:
            raise ValueError(
                f"The argument `hook` does not support type {type(hook)}, "
                f"it should be given a type with the one of class, list, dict."
            )
        for fn in funcs:
            if self.is_hook_event(fn):
                self.replace_hook_event(fn, self.wrap_fn(fn))

    def is_hook_event(self, name: str):
        return name.startswith("on") and (
            name.endswith("start") or name.endswith("end")
        )

    @abstractmethod
    def wrap_fn(self, event_name: str):
        pass
