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

from typing import List

import torch
from ixrt.deploy.api import create_executor
from ixrt.deploy.ir.graph import Graph
from ixrt.deploy.ir.operator_type import OperatorType as OP
from torch import nn
from torch.nn import Parameter

from ..quantizer_config import QuantizerConfig, get_default_quant_operator_config
from ..static_quantizer import PostTrainingStaticQuantizer


def convert_tensor_to_parameter(graph):
    for var in graph.variables.values():
        if var.is_parameter and torch.is_tensor(var.value):
            var.value = torch.nn.Parameter(var.value)
        elif var.is_parameter:
            print(
                f"Warning: Skip convert {var.name} to parameter, because it is not a torch.Tensor."
            )
    return graph


class TrainableGraph(nn.Module):
    def __init__(self, graph: Graph, executor=None, qat=True):
        super(TrainableGraph, self).__init__()
        self.graph = convert_tensor_to_parameter(graph)
        self.executor = create_executor("torch") if executor is None else executor
        self._init_parameters()
        self.qat = qat

    def forward(self, inputs):
        if self.qat:
            exec_ctx = self.executor.enable_quant_context
        else:
            exec_ctx = self.executor.disable_quant_context
        with exec_ctx():
            origin_quant_output = self.executor.enable_quant_output
            self.executor.enable_quant_output = False
            out = self.executor.execute_graph(self.graph, inputs)
            self.executor.enable_quant_output = origin_quant_output
            return out

    def _init_parameters(self):
        for var in self.graph.variables.values():
            if var.is_parameter and torch.is_tensor(var.value):
                var.value = torch.nn.Parameter(var.value)
                self.register_parameter(var.name, var.value)
            elif var.is_parameter:
                print(
                    f"Warning: Skip convert {var.name} to parameter, because it is not a torch.Tensor."
                )


def create_default_qconfig():
    default_config = get_default_quant_operator_config()
    qconfig = QuantizerConfig(use_qat=True)

    qconfig.operator_config.set_global_config(None)
    qconfig.operator_config.set_config_with_op(
        OP.CONV, default_config.get_global_config()
    )
    qconfig.operator_config.set_config_with_op(
        OP.GEMM, default_config.get_global_config()
    )
    qconfig.operator_config.set_config_with_op(
        OP.CONV_TRANSPOSE, default_config.get_config_with_op(OP.CONV_TRANSPOSE)
    )
    return qconfig


def convert_to_qat(
    graph, calibration_dataloader, qconfig=None, preprocess=None, executor=None
):
    if qconfig is None:
        qconfig = create_default_qconfig()

    if executor is None:
        executor = create_executor("torch")

    static_quantizer = PostTrainingStaticQuantizer(
        calibration_dataloader,
        executor=executor,
        qconfig=qconfig,
        preprocess=preprocess,
    )

    converted_graph = static_quantizer(graph)
    return TrainableGraph(converted_graph, executor=executor, qat=True)
