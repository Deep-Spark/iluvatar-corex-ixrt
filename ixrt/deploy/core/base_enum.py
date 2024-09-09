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

from enum import Enum


class BaseEnum(Enum):

    F = False

    @classmethod
    def from_value(cls, value):
        for item in cls.__dict__.values():
            if isinstance(item, cls) and item.value == value:
                return item
        raise RuntimeError(f"Not found {value} in {cls}.")


if __name__ == "__main__":
    print(BaseEnum.from_value(False))
