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

import json
import sys
from argparse import REMAINDER

from ixrt.deploy.autotuning.tuner import AutotuningEvaluator, CommandLineTuner

from .base import CLI_REGISTRY, BaseCLI


@CLI_REGISTRY.registe()
class TunerCLI(BaseCLI):
    def predefine_args(self):
        self.parser.add_argument(
            "config",
            type=str,
            help="The config path of autotuning, "
            "it should contain hyperparameters "
            "and arguments of parsing log.",
        )
        self.parser.add_argument("script", type=str, help="Tuning script.")
        self.parser.add_argument(
            "script_args", nargs=REMAINDER, help="The arguments of script."
        )

    def command_name(self):
        return "tune"

    def run(self, args):
        config_path = args.config
        with open(config_path) as f:
            config = json.load(f)

        tuner = CommandLineTuner(
            log_parser_args=config.get("parser", {}),
            popen_args=config.get("popen", {}),
            evaluator=AutotuningEvaluator(**config.get("evaluator", {})),
        )

        tuner.tune(command=self.get_tuning_script(args), space=config.get("tuner", {}))

    def get_tuning_script(self, args):
        if args.script.endswith(".py"):
            args.script = f"{sys.executable} {args.script}"

        script = [args.script] + args.script_args
        return " ".join(script)
