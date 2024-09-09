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

import importlib
from typing import Union

from ixrt.deploy.ops import reorg

EXTENSION_OPS = ["dcn"]


def load_operator(name: str):
    importlib.import_module(f"ixrt.deploy.ops.{name}")


def enable_operator_extensions(ops: Union[str, list] = "all"):
    if ops == "all":
        for mod in EXTENSION_OPS:
            if isinstance(mod, str):
                load_operator(mod)
            else:
                load_operator(mod[-1])
        return

    if isinstance(ops, str):
        ops = [ops]

    for mod in EXTENSION_OPS:
        if isinstance(mod, str):
            if mod in ops:
                load_operator(mod)
        else:
            if mod[0] in ops:
                load_operator(mod[-1])
