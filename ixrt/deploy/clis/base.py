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


from abc import abstractmethod
from argparse import ArgumentParser, Namespace

from ixrt.deploy.core import Registry

CLI_REGISTRY = Registry("Cli Register")


class BaseCLI:
    def __init__(self, parser=None, *args, **kwargs):
        if parser is None:
            self.parser = ArgumentParser(description=self.description, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        cli_args = self.parse_args(*args, **kwargs)
        self.run(cli_args)

    @property
    def description(self):
        return None

    @abstractmethod
    def command_name(self):
        pass

    def predefine_args(self):
        pass

    def parse_args(self, *args, **kwargs) -> Namespace:
        self.predefine_args()
        args, other = self.parser.parse_known_args(*args, **kwargs)
        if other:
            msg = _("unrecognized arguments: %s")
            print(msg % " ".join(other))
        return args

    @abstractmethod
    def run(self, args: Namespace):
        pass
