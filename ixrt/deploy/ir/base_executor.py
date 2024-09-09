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

import copy
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Mapping, Optional, Union

import torch

from ..core import Registry
from ..core.hook import PriorityHooks
from .data_type import VariableType
from .executor_hook import ExecutorHook
from .graph import Graph
from .operator import Operator
from .operator_type import SKIP_SIMULATIVE_QUANT_OPERATORS
from .utils import eliminate_var_of_attr

InputsType = Union[dict, Any]


class StopExecGraphException(Exception):
    pass


class ExecuteGraphMix(object):
    def init_execute_graph(self, operator_register: Registry):
        self.hooks: ExecutorHook = PriorityHooks()
        self.hooks.add_hook(ExecutorHook())
        self.operator_register = operator_register

        self._current_graph: Optional[Graph] = None
        self._enable_quant = False

    @property
    def enable_qaunt(self):
        return self._enable_quant

    @enable_qaunt.setter
    def enable_qaunt(self, enable):
        self._enable_quant = enable

    @contextmanager
    def enable_quant_context(self):
        origin_status = self.enable_qaunt
        self.enable_qaunt = True
        yield self
        self.enable_qaunt = origin_status

    @contextmanager
    def disable_quant_context(self):
        origin_status = self.enable_qaunt
        self.enable_qaunt = False
        yield self
        self.enable_qaunt = origin_status

    def execute_graph(self, graph: Graph, inputs: InputsType):
        try:
            return self._execute_graph(graph, inputs)
        except StopExecGraphException:
            return

    def _execute_graph(self, graph: Graph, inputs: InputsType):
        self._current_graph = graph
        self._set_hook_state(graph)
        self.set_inputs(graph, inputs)
        sorted_ops = graph.toposort()
        self.hooks.on_exec_graph_start()
        for op in sorted_ops:
            self.hooks.on_exec_operator_start(op)
            output = self.exec_operator(graph, op)
            self.set_output(graph, op, output)
            self.hooks.on_exec_operator_end(op, output)
        self.hooks.on_exec_graph_end()
        return self.get_outputs(graph)

    def exec_operator(self, graph: Graph, op: Operator):
        if self.enable_qaunt and op.op_type not in SKIP_SIMULATIVE_QUANT_OPERATORS:
            return self.exec_quant_operator(graph, op)
        return self.exec_nonquant_operator(graph, op)

    def exec_nonquant_operator(self, graph: Graph, op: Operator):
        inputs = self._get_op_var_values(op.inputs)
        attr = self.get_operator_attr(graph, op)
        op_func = self.get_operator(op)
        return op_func(self, op, inputs, attr)

    def exec_quant_operator(self, graph: Graph, op: Operator):
        raise NotImplementedError()

    def get_operator(self, op: Operator):
        if not self.operator_register.containe(op.op_type):
            raise RuntimeError(f"Not support {op.op_type}.")
        return self.operator_register.get(op.op_type)

    def get_operator_attr(self, graph: Graph, op: Operator):
        return eliminate_var_of_attr(graph, op.attributes)

    def set_inputs(self, graph: Graph, inputs: InputsType):
        if len(graph.inputs) == 1 and not isinstance(inputs, Mapping):
            if isinstance(inputs, (tuple, list)) and len(inputs) == 1:
                graph.first_input.value = inputs[0]
            else:
                graph.first_input.value = inputs
            return

        inputs_key = graph.input_names
        if isinstance(inputs, (tuple, list)):
            inputs = dict(zip(inputs_key, inputs))

        for input_key in inputs:
            graph.inputs[input_key].value = inputs[input_key]

    def set_output(self, graph: Graph, op: Operator, outputs):
        if outputs is None:
            return

        output_keys = op.outputs
        if len(output_keys) == 1:
            graph.set_var_value(output_keys[0], outputs)
        else:
            if len(output_keys) != len(outputs):
                raise RuntimeError(
                    f"Cannot set output of {op.name}({op.op_type}), "
                    f"because the number of outputs is not same, "
                    f"got {len(output_keys)} and {len(outputs)}."
                )

            for var, value in zip(output_keys, outputs):
                graph.set_var_value(var, value)

    def remove_outputs_data(self):
        graph = self._current_graph
        outputs = graph.outputs
        sorted_ops = graph.toposort()
        for op in sorted_ops:
            output_keys = op.outputs
            for key in output_keys:
                value = graph.get_var_value(key)
                if isinstance(value, torch.Tensor):
                    graph.delete_variable(key)

    def get_outputs(self, graph: Graph):
        outputs = graph.outputs
        if len(outputs) == 1:
            _outputs = list(outputs.values())[0]
            if _outputs.var_type == VariableType.MAP:
                _outputs = _outputs.value
                output_values = OrderedDict()
                for name in _outputs:
                    output_values[name] = graph.get_var_value(_outputs[name])
                return output_values

            elif _outputs.var_type == VariableType.LIST:
                _outputs = _outputs.value
                output_values = []
                for output in _outputs:
                    output_values.append(graph.get_var_value(output))
                return output_values

        outputs = OrderedDict()
        for output in graph.outputs:
            outputs[output] = graph.get_var_value(output)

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs

    def add_hook(self, hook, priority=1):
        self.hooks.add_hook(hook, priority)

    def remove_hook(self, hook):
        self.hooks.remove_hook(hook)

    def _set_hook_state(self, graph):
        for hook in self.hooks.hooks.values():
            hook = hook.hooks
            for _hook in hook:
                _hook.set_graph(graph)
                _hook.set_executor(self)

    def _get_op_var_values(self, names: Union[list, dict]):
        def map_list(_names: list):
            _outs = []
            for _name in _names:
                if isinstance(_name, str):
                    _outs.append(self._current_graph.get_var_value(_name))
                elif isinstance(_name, list):
                    _outs.append(map_list(_name))
                elif isinstance(_name, Mapping):
                    _outs.append(map_dict(_name))
                else:
                    _outs.append(_name)
            return _outs

        def map_dict(_names: Mapping):
            _outs = dict()
            for _name, _value in _names.items():
                if isinstance(_value, str):
                    _outs[_name] = self._current_graph.get_var_value(_value)
                elif isinstance(_name, list):
                    _outs[_name] = map_list(_value)
                elif isinstance(_name, Mapping):
                    _outs[_name] = map_dict(_value)
                else:
                    _outs[_name] = _value
            return _outs

        if isinstance(names, list):
            return map_list(names)

        if isinstance(names, Mapping):
            return map_dict(names)

        raise RuntimeError(f"Fetch the value of variable fail, got names = {names}.")


class BaseExecutor(ExecuteGraphMix):
    def __init__(self, operator_register: Registry):
        self.init_execute_graph(operator_register)
