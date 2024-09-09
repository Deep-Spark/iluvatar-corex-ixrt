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
from typing import Type

from ixrt.deploy.core import BaseDataClass


@dataclass()
class BaseOperatorAttr(BaseDataClass):
    def __post_init__(self):
        for name, val in self.to_dict().items():
            if isinstance(val, bytes):
                setattr(self, name, val.decode("utf8"))
