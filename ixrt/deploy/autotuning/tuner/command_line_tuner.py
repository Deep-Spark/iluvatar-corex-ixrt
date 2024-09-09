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

from typing import Callable, Mapping, Union

from ixrt.deploy.utils.popen import Popen

from ..log_parser import LogParser
from ..paramter import (
    ContinuousParameter,
    DiscreteParameter,
    ParameterSpace,
    create_parameter_space,
)
from .tuner import Tuner


class CommandLineTuner(Tuner):
    def __init__(self, log_parser_args: dict, popen_args: dict = None, *args, **kwargs):
        super(CommandLineTuner, self).__init__(*args, **kwargs)
        self.log_parser_args = log_parser_args
        self.log_parser = LogParser(**log_parser_args)
        self.popen_args = popen_args or dict()

    def tune(self, command, space: Union[ParameterSpace, Mapping]):
        if isinstance(space, Mapping):
            space = self.create_parameter_space(space)
        target_fn = self.open_cmdline_process(command)
        super(CommandLineTuner, self).tune(target_fn, space)

    def open_cmdline_process(self, command) -> Callable:
        def target_fn(**kwargs):
            hyper_params = command
            for name, value in kwargs.items():
                if name.startswith("__"):
                    hyper_params += f" {value}"
                else:
                    hyper_params += f" {name} {value}"
            print("Run:", hyper_params)
            output = Popen.get_output(hyper_params, **self.popen_args)
            metrics = self.log_parser.parse(output)
            if len(metrics) == 0:
                metrics = "N/A"
            if isinstance(metrics, (tuple, list)):
                return metrics[-1]
            return metrics

        return target_fn

    def create_parameter_space(self, space: dict):
        space_name = "grid"
        if "space" in space:
            space_name = space.pop("space")

        param_list = []
        for name, value in space.items():
            param_list.append(self.create_parameter(name, value))
        return create_parameter_space(space_name, param_list=param_list)

    def create_parameter(self, name, value):
        if isinstance(value, (tuple, list)):
            return DiscreteParameter(name, value)
        elif isinstance(value, (int, float)):
            return DiscreteParameter(name, [value])
        elif isinstance(value, str):
            if "," in value:
                return DiscreteParameter(name, value.split(","))
            elif ":" in value:
                start, end, num = value.split(":")
                start = float(start)
                end = float(end)
                num = int(num)
                return ContinuousParameter(name, start, end, num)

        return DiscreteParameter(name, value)
