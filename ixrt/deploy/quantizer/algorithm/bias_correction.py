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

from collections import defaultdict

import torch.nn
from ixrt.deploy.core.progress_bar import progress_bar
from ixrt.deploy.ir import BaseExecutor, Graph
from ixrt.deploy.ir.executor_hook import LambdaExecutorHook
from ixrt.deploy.ir.operator_type import OperatorType as OP

BIAS_CORRECTION_OPS = [OP.CONV, OP.GEMM, OP.CONV_TRANSPOSE]


class BiasCorrection(object):
    def __init__(self, graph: Graph, executor: BaseExecutor):
        self.graph = graph
        self.executor = executor
        self.bias_error_table = defaultdict(int)
        self.call_operator_count = defaultdict(int)
        self._collect_error_hook = None

    def start_collect_quant_error(self):
        self._collect_error_hook = LambdaExecutorHook(
            LambdaExecutorHook.on_exec_operator_end, self.collect_bias_error
        )
        self.executor.add_hook(hook=self._collect_error_hook)

    def collect_bias_error(self, operator, outputs):
        if operator.op_type not in BIAS_CORRECTION_OPS:
            return

        if len(operator.inputs) < 3:
            return

        if self.executor.enable_qaunt:
            quant_out = outputs
            fp_out = self.executor.exec_nonquant_operator(self.graph, operator)
        else:
            fp_out = outputs
            quant_out = self.executor.exec_quant_operator(self.graph, operator)

        quant_error = fp_out - quant_out

        quant_error = self.reduce_tensor_error(operator, quant_error).detach().cpu()
        self.bias_error_table[operator.name] += quant_error
        self.call_operator_count[operator.name] += 1

    def reduce_tensor_error(self, operator, error):
        ndim = error.ndim
        keepdim = 1
        if operator.op_type == OP.GEMM:
            keepdim = -1

        reduction_dims = list(range(ndim))
        reduction_dims.pop(keepdim)
        error = error.mean(dim=reduction_dims)
        return error

    def correct(self):
        self.executor.remove_hook(self._collect_error_hook)
        progress = progress_bar(self.bias_error_table.items(), desc="BiasCorrection")
        for op_name, error in progress:
            operator = self.graph.get_operator(op_name)
            if len(operator.inputs) > 2:
                error = error / self.call_operator_count[op_name]
                new_bias = self.graph.get_var_value(operator.inputs[2])
                try:
                    new_bias = new_bias + error.to(new_bias.device)
                except Exception as ex:
                    print("Operator:", operator.name)
                    print(
                        "Output shape:",
                        self.graph.get_variable(operator.outputs[0]).value.shape,
                    )
                    print("Bias shape:", new_bias.shape)
                    print("Error shape:", error.shape)
                    raise ex
                self.graph.set_var_value(
                    operator.inputs[2], torch.nn.Parameter(new_bias)
                )
