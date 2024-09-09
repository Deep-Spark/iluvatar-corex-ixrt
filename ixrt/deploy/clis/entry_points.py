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

# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

from . import CLI_REGISTRY, BaseCLI

_clis = []
for cli_cls in CLI_REGISTRY.handlers.values():
    _cli_cmd: BaseCLI = cli_cls()
    _cli_name = _cli_cmd.command_name()
    _clis.append(_cli_name)
    globals()[_cli_name] = _cli_cmd


def make_execute_path():
    preffix = "ixrt.deploy.clis.entry_points"
    clis = []
    for cli_name in _clis:
        clis.append(f"ixrt-{cli_name}={preffix}:{cli_name}")

    return clis
