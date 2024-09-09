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

import copy
from dataclasses import dataclass
from typing import Iterable


@dataclass()
class BaseDataClass(object):
    def to_dict(self, *args, **kwargs):
        fields = self.get_fields()
        ret = dict()
        for field in fields:
            ret[field] = getattr(self, field)
        return ret

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def get_fields(cls) -> Iterable:
        return cls.data_fields().keys()

    @classmethod
    def data_fields(cls) -> dict:
        return cls.__dataclass_fields__

    def get(self, k, default=None):
        val = getattr(self, k, default)
        if val is None:
            return default
        return val
