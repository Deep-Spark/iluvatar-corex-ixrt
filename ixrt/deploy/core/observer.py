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

import weakref
from typing import Any, Callable

from ixrt.deploy.utils.object import isimmutable_var


class Oversable(object):
    def __init__(self):
        self.watched_variable = None

    def set_watched_variable(self, var):
        if isimmutable_var(var):
            self.watched_variable = var
        else:
            self.watched_variable = weakref.ref(var)


class PropertyObserver(Oversable):
    def change_before(self, value):
        pass

    def change_after(self, value):
        pass


class LambdaPropertyObserver(PropertyObserver):
    def __init__(
        self,
        change_before: Callable[[Any], None] = None,
        change_after: Callable[[Any], None] = None,
    ):
        super(LambdaPropertyObserver, self).__init__()
        self._change_before = change_before
        self._change_after = change_after

    def change_before(self, value):
        if self._change_before is not None:
            self._change_before(value)

    def change_after(self, value):
        if self._change_after is not None:
            self._change_after(value)
