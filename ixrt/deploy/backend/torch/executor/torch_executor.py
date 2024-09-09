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

import warnings
from typing import Mapping

import numpy as np
import torch
from ixrt.deploy.core.producer_type import ProducerType
from ixrt.deploy.ir import BaseExecutor, Graph, Operator
from ixrt.deploy.ir.executor_hook import ExecutorHook
from ixrt.deploy.quantizer import quant_function as QF

from .operators import TORCH_EXECUTOR_OPERATORS
from .quant_parameter_tensor import QuantParameterTensor


class QuantParamsToTensorHook(ExecutorHook):
    def on_exec_graph_start(self):
        for name, qparam in self.graph.quant_parameters.items():
            if not isinstance(qparam, QuantParameterTensor):
                self.graph.quant_parameters[
                    name
                ] = QuantParameterTensor.from_quant_param(qparam)


def to_device(graph: Graph, device):
    graph.device = device
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
        except:
            warnings.warn("Cannot set device.")

    def ndarr_to_tensor(value):
        return torch.from_numpy(value).to(device)

    def list_to_tensor(value):
        for idx, v in enumerate(value):
            value[idx] = var_to_tensor(v)
        return value

    def dict_to_tensor(value):
        for k, v in value.items():
            value[k] = var_to_tensor(v)
        return value

    def var_to_tensor(value):
        if isinstance(value, np.ndarray):
            return ndarr_to_tensor(value)
        elif isinstance(value, Mapping):
            return dict_to_tensor(value)
        elif isinstance(value, (list, tuple)):
            return list_to_tensor(value)
        elif torch.is_tensor(value):
            return value.to(device)
        return value

    for var in graph.variables.values():
        var.value = var_to_tensor(var.value)

    return graph


class IrGraphToDevice(ExecutorHook):
    def on_exec_graph_start(self):
        self.executor: "TorchExecutor"

        if (
            self.graph.device == self.executor.default_device()
            and self.graph.device is not None
        ):
            return

        if len(self.graph.producers) > 0 and self.graph.producers[0] in [
            ProducerType.FX,
            ProducerType.TORCH_JIT,
        ]:
            return

        to_device(self.graph, self.executor.default_device())


class QuantOutputsHook(ExecutorHook):
    def __init__(self, enable=True):
        super(QuantOutputsHook, self).__init__()
        self.enable = enable

    def on_exec_graph_end(self):
        if not self.enable:
            return

        if not self.executor.enable_qaunt:
            return

        for out in self.graph.outputs:
            if self.graph.is_quant_variable(out):
                out = self.graph.get_variable(out)
                qparam = self.graph.get_quant_parameter(out)
                qparam = qparam.to(out.value.device)
                qout = QF.fake_linear_quantizer(out.value, qparam)
                out.value = qout


class TorchExecutor(BaseExecutor):
    def __init__(self):
        super(TorchExecutor, self).__init__(operator_register=TORCH_EXECUTOR_OPERATORS)

        default_priority = 99999

        self.add_hook(IrGraphToDevice(), priority=default_priority)
        self.add_hook(QuantParamsToTensorHook(), priority=default_priority - 1)
        self.quant_output_hook = QuantOutputsHook()
        self.add_hook(self.quant_output_hook, priority=default_priority - 2)
        self._device = None

    @property
    def enable_quant_output(self):
        return self.quant_output_hook.enable

    @enable_quant_output.setter
    def enable_quant_output(self, enable):
        self.quant_output_hook.enable = enable

    def set_device(self, device):
        self._device = device
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

    @property
    def device(self):
        return self._device

    def default_device(self):
        if self._device is not None:
            return self.device
        return 0 if torch.cuda.is_available() else "cpu"

    def exec_quant_operator(self, graph: Graph, op: Operator):
        inputs = self._get_op_var_values(op.inputs)

        device = self.default_device()
        for x in inputs:
            if torch.is_tensor(x):
                device = x.device
                break

        for idx, op_input in enumerate(op.inputs):
            if graph.is_quant_variable(op_input):
                qparam = graph.get_quant_parameter(op_input)
                qparam = qparam.to(device)
                inputs[idx] = QF.fake_linear_quantizer(inputs[idx], qparam)

        attr = self.get_operator_attr(graph, op)
        op_func = self.get_operator(op)
        return op_func(self, op, inputs, attr)
