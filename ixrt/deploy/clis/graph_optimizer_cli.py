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

from argparse import Namespace

from ..api import (
    LoadQuantParamtersPPQStype,
    PassSequence,
    Pipeline,
    create_passes,
    create_source,
    create_target,
)
from ..utils import append_name_in_file_path
from .base import CLI_REGISTRY, BaseCLI


@CLI_REGISTRY.registe()
class GraphOptimizerCli(BaseCLI):
    @property
    def description(self):
        return "This command can optimize graph, such as fusing operators, formating operator, clearing unused variables, ..."

    def command_name(self):
        return "optimize"

    def predefine_args(self):
        self.parser.add_argument(
            "--model", type=str, required=True, help="File path of model"
        )
        self.parser.add_argument(
            "--quant_file", type=str, help="File path of quantization parameter"
        )
        self.parser.add_argument(
            "--saved_model", type=str, default=None, help="Saved file path of model "
        )
        self.parser.add_argument(
            "--saved_quant_file",
            type=str,
            default=None,
            help="Saved file path of quantization parameter",
        )
        self.parser.add_argument(
            "--passes",
            type=str,
            default="default",
            help="The sequence of passes by comma to split.",
        )
        self.parser.add_argument(
            "--backend",
            type=str,
            default=None,
            help="Convert the tensors of graph to backend, support torch now.",
        )

    def run(self, args: Namespace):
        if args.saved_model is None:
            args.saved_model = append_name_in_file_path(args.model, "-optimized")

        if args.saved_quant_file is None and args.quant_file is not None:
            args.saved_quant_file = append_name_in_file_path(
                args.quant_file, "-optimized"
            )

        pipeline = []

        if args.quant_file is not None:
            pipeline.append(LoadQuantParamtersPPQStype(args.quant_file))

        if args.backend == "torch":
            from ixrt.deploy.api.pipeline import ToDevice

            pipeline.append(ToDevice("cpu"))

        pipeline.extend(
            [
                PassSequence(*create_passes(args.passes.split(","))),
                create_target(
                    args.saved_model, quant_params_path=args.saved_quant_file
                ),
            ]
        )
        pipeline = Pipeline(create_source(args.model), *pipeline)
        pipeline.run()
