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
import copy
import os
import subprocess
from subprocess import Popen as _Popen
from typing import Callable, List, Union

from .real_tempfile import TemporaryFile


def create_subproc_env():
    env = copy.copy(os.environ)
    env["IXDEPLOY"] = "1"
    return env


ReturnCode = int


def get_output_with_pipe(
    command, shell=None, callback: Callable[[list], None] = None, *args, **kwargs
):
    if shell is None:
        shell = isinstance(command, str)

    if shell and not isinstance(command, str):
        command = " ".join(command)

    stream = subprocess.Popen(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        *args,
        **kwargs
    )
    outputs = []
    while 1:
        exit_code = stream.poll()
        if exit_code is None:
            if stream.stdout.readable():
                outputs.append(stream.stdout.readline().decode("utf8").rstrip())
                if callback is not None:
                    callback(outputs[-1:])
                if outputs[-1] in ["", "\n"]:
                    continue
                print(outputs[-1])
        else:
            if stream.stdout.readable():
                lines = stream.stdout.readlines()
                lines = [line.decode("utf8".rstrip()) for line in lines]
                outputs.extend(lines)
                if callback is not None:
                    callback(outputs[-1:])
                print("\n".join(lines))
            break

    return outputs


def get_output_with_tempfile(command, *args, **kwargs):
    if not isinstance(command, (list, tuple)):
        command = [command]
    stdout = None
    with TemporaryFile(with_open=True) as file:
        command.extend(["|", "tee", file.name])
        command = " ".join(command)

        res = subprocess.run(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            shell=True,
            *args,
            **kwargs
        )
        output = file.readlines()

    return output


def execute_shell(command, *args, **kwargs):
    if "env" not in kwargs:
        kwargs["env"] = create_subproc_env()

    if not isinstance(command, (list, tuple)):
        command = [command]

    command = " ".join(command)
    res = subprocess.run(command, shell=True, *args, **kwargs)
    return res


class Popen:
    def __init__(self, cmd: Union[str, list], **kwargs):
        if isinstance(cmd, (list, tuple)):
            cmd = " ".join(cmd)
        self.cmd = cmd
        self._proc: _Popen = None
        self.proc_args = kwargs

    @property
    def proc(self):
        return self._proc

    def start(self):
        self._proc = _Popen(self.cmd, shell=True, **self.proc_args)

    def wait(self) -> ReturnCode:
        return self._proc.wait()

    def kill(self):
        self._proc.terminate()
        return self._proc.kill()

    @staticmethod
    def get_output(
        command: Union[List, str], capture_method: str = "tempfile", *args, **kwargs
    ):
        if "env" not in kwargs:
            kwargs["env"] = create_subproc_env()

        if capture_method == "tempfile":
            return get_output_with_tempfile(command, *args, **kwargs)
        return get_output_with_pipe(command, *args, **kwargs)
