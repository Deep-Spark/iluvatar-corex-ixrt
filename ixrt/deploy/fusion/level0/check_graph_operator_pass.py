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

from ixrt.deploy.ir import Graph
from ixrt.deploy.ir import OperatorType as OP

from ..base_pass import BasePass, registe_pass

INVALID_NAME = [""]
SKIP_CHECK_OPERATORS = [OP.CLIP]


@registe_pass(level=0)
class CheckGraphOperatorPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.invalid_operators = []
        self.invalid_inputs = []
        map(self.check_inputs_outputs, graph.operators.values())
        self.summary()
        return graph

    def check_inputs_outputs(self, operator):
        if operator.op_type in SKIP_CHECK_OPERATORS:
            return

        if operator.name in INVALID_NAME:
            self.invalid_operators.append(operator)

        for variable in operator.inputs + operator.outputs:
            if variable in INVALID_NAME:
                self.invalid_inputs.append(operator)

    def summary(self):
        if len(self.invalid_inputs) == 0 or len(self.invalid_operators) == 0:
            return

        error_msg = ""

        if len(self.invalid_inputs) != 0:
            error_msg = "Got illegal the inputs of operator:", [
                str(op) for op in self.invalid_inputs
            ]

        if len(self.invalid_operators) != 0:
            error_msg += "\n" + "Got illegal operator name:", [
                str(op) for op in self.invalid_operators
            ]

        raise RuntimeError(error_msg)
