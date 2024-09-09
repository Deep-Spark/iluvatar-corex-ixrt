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
import warnings
from typing import Dict, List, Mapping, Optional, Union

try:
    from typing import OrderedDict as OrderedDictType
except:
    from typing import Dict as OrderedDictType

from collections import OrderedDict

from ixrt.deploy.utils.object import flatten_container

from ..core.producer_type import ProducerType
from .data_type import VariableType
from .operator import Operator
from .variable import Placeholder, Variable


class Graph(object):
    def __init__(self, meta=None):
        self.meta = meta
        self.producers: List[ProducerType] = []
        self.device = None
        self._operators: OrderedDictType[str, Operator] = OrderedDict()
        self._variables: OrderedDictType[str, Variable] = OrderedDict()

        self._inputs: OrderedDictType[str, Variable] = OrderedDict()
        self._outputs: OrderedDictType[str, Variable] = OrderedDict()

        self._quant_parameters: OrderedDictType = OrderedDict()

    @property
    def operators(self) -> Dict[str, Operator]:
        return self._operators

    @property
    def variables(self) -> Dict[str, Variable]:
        return self._variables

    def set_meta(self, meta):
        self.meta = meta

    def containe_var(self, var: Union[str, Variable]):
        if isinstance(var, Variable):
            var = var.name
        return var in self.variables

    def containe_operator(self, op: Union[str, Operator]):
        if isinstance(op, Operator):
            op = op.name
        return op in self.operators

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_names(self):
        return list(self._inputs.keys())

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_names(self):
        return list(self._outputs.keys())

    @property
    def quant_parameters(self):
        return self._quant_parameters

    def clear_quant_parameters(self):
        self.quant_parameters.clear()
        for var in self.variables.values():
            var.remove_value_observer()

    @property
    def first_input(self) -> Variable:
        return list(self._inputs.values())[0]

    def is_leaf_variable(self, var):
        if isinstance(var, str):
            var = self.get_variable(var)
        src_op = self.get_src_operator(var)
        return src_op is None or var.is_parameter

    def is_quant_variable(self, var: Union[str, Variable]):
        if isinstance(var, Variable):
            var = var.name
        return var in self.quant_parameters

    def add_quant_parameter(self, name, params):
        self.quant_parameters[name] = params

    def delete_quant_parameter(self, var):
        if isinstance(var, Variable):
            var = var.name
        if var in self.quant_parameters:
            return self.quant_parameters.pop(var)

    def get_quant_parameter(self, var: Union[str, Variable], **kwargs):
        if isinstance(var, Variable):
            var = var.name
        if self.is_quant_variable(var):
            return self.quant_parameters[var]
        if "default" in kwargs:
            return kwargs["default"]
        raise RuntimeError(f"Not found quantized parameter `{var}`.")

    def add_input(self, var: Variable):
        if var.name not in self.variables:
            self.add_variable(var)
        self._inputs[var.name] = var

    def add_output(self, var: Union[str, Variable]):
        if isinstance(var, str) and not self.containe_var(var):
            raise RuntimeError(
                f"The output variable must be exist in graph, got {var}."
            )

        if isinstance(var, str):
            var = self.get_variable(var)

        if not self.containe_var(var):
            self.add_variable(var)

        self._outputs[var.name] = var

    def add_variable(self, var: Variable):
        self.variables[var.name] = var

    def delete_variable(self, var: Union[str, Variable]) -> Variable:
        if isinstance(var, Variable):
            var = var.name

        if var in self.variables:
            return self.variables.pop(var)

    def rename_operator(self, old_name, new_name):
        if old_name == new_name:
            return

        op = self.get_operator(old_name)
        op.name = new_name

        self._operators.pop(old_name)
        self._operators[new_name] = op

    def rename_vaiable(
        self,
        old_name,
        new_name,
        with_variables: bool = True,
        with_operator_outputs: bool = False,
    ):
        if new_name == old_name:
            return

        for input in self.input_names:
            old_var = self.get_variable(input)
            if self.containe_var(input) and self.is_leaf_variable(old_var.name):
                continue

            if self.containe_var(input) and old_var.name == old_name:
                old_var.name = new_name

                self.inputs.pop(old_name)
                self.inputs[new_name] = old_var

        for out in self.output_names:
            if self.containe_var(out) and self.get_variable(out).name == old_name:
                var = self.get_variable(out)
                var.name = new_name

                self.outputs.pop(old_name)
                self.outputs[new_name] = var

        for op in self.operators.values():
            for idx in range(len(op.inputs)):
                if op.inputs[idx] == old_name:
                    op.inputs[idx] = new_name

            if with_operator_outputs:
                for idx in range(len(op.outputs)):
                    if op.outputs[idx] == old_name:
                        op.outputs[idx] = new_name

        if not with_variables:
            return

        var = self.get_variable(old_name)
        var.name = new_name

    def set_var_value(self, var: Union[str, Variable], value):
        if isinstance(var, Variable):
            var = var.name
        self.variables[var].value = value

    def get_var_value(self, var: Union[str, Variable]):
        if isinstance(var, Variable):
            var = var.name

        var_value = self.variables[var].value

        def _transform_var_to_value(_var, key):
            _value = _var[key]
            if isinstance(_value, Variable):
                _var[key] = _value.value
            elif isinstance(_value, (tuple, list)):
                _var[key] = list(_value)
                _trans_value_list(_var[key])
            elif isinstance(_value, dict):
                _var[key] = dict(_value)
                _trans_value_dict(_var[key])

        def _trans_value_list(_var: list):
            for key, v in enumerate(_var):
                _transform_var_to_value(_var, key)

        def _trans_value_dict(_var: dict):
            for key in _var:
                _transform_var_to_value(_var, key)

        if isinstance(var_value, (tuple, list)):
            var_value = list(var_value)
            _trans_value_list(var_value)
        elif isinstance(var_value, dict):
            var_value = dict(var_value)
            _trans_value_dict(var_value)
        elif isinstance(var_value, Variable):
            var_value = var_value.value

        return var_value

    def get_var_name_from_value(self, value) -> Optional[str]:
        for name, var in self.variables.items():
            if var.value is value:
                return name
        return None

    def clear_var_value(self):
        for var in self.variables.values():
            var.value = None

    def add_operator(self, op: Operator):
        self.operators[op.name] = op

        for out in op.outputs:
            if not self.containe_var(out):
                self.add_variable(Placeholder(out))

    def delete_operator(self, op: Union[str, Operator]):
        if isinstance(op, Operator):
            op = op.name

        if op in self.operators:
            return self.operators.pop(op)

    def get_operator(self, name: str) -> Operator:
        return self.operators[name]

    def get_variable(self, name: str) -> Variable:
        return self.variables[name]

    def add_operator_input(self, operator: Union[str, Operator], var: Variable):
        if isinstance(operator, str):
            operator = self.get_operator(operator)
        self.add_variable(var)
        operator.inputs.append(var.name)

    def get_operator_input_vars(self, op: Union[str, Operator]) -> Dict[str, Variable]:
        if isinstance(op, str):
            op = self.get_operator(op)

        inputs = dict()
        for name in op.inputs:
            inputs[name] = self.get_variable(name)

        return inputs

    def get_operator_output_var(self, op: Union[str, Operator]) -> Dict[str, Variable]:
        if isinstance(op, str):
            op = self.get_operator(op)

        outputs = dict()
        for name in op.outputs:
            outputs[name] = self.get_variable(name)

        return outputs

    def get_src_operator(self, var: Union[str, Variable]) -> Optional[Operator]:
        """
        Graph: src_op -> var
        """
        if isinstance(var, Variable):
            var = var.name

        for op in self.operators.values():
            if var in op.outputs:
                return op

        return None

    def get_dst_operators(
        self, var: Union[str, Variable], with_attr_var=False
    ) -> List[Operator]:
        """
        Graph: src_op -> var -> dst_op1
                          | --> dst_op2
        """
        if isinstance(var, Variable):
            var = var.name

        dst_ops = []
        for op in self.operators.values():
            op_inputs = flatten_container(op.inputs)
            if with_attr_var:
                op_inputs.extend(self.get_variables_of_attr(op.attributes))
            if var in op_inputs:
                dst_ops.append(op)

        return dst_ops

    def get_previous_operators(
        self, op: Union[str, Operator], with_attr_var=False
    ) -> List[Operator]:
        """
        Graph: previous_ops -> op -> next_ops
        """
        if isinstance(op, str):
            op = self.get_operator(op)

        inputs = copy.deepcopy(op.inputs)
        if with_attr_var:
            inputs.extend(self.get_variables_of_attr(op.attributes))

        inputs = flatten_container(inputs)

        prev_ops = []
        for input_name in inputs:
            prev_op = self.get_src_operator(input_name)
            if prev_op is not None:
                prev_ops.append(prev_op)

        return prev_ops

    def get_next_operators(
        self, op: Union[str, Operator], with_attr_var=False
    ) -> List[Operator]:
        """
        Graph: previous_ops -> op -> next_ops
        """
        if isinstance(op, str):
            op = self.get_operator(op)

        outputs = flatten_container(op.outputs)

        next_ops = []
        for output_name in outputs:
            next_ops.extend(
                self.get_dst_operators(output_name, with_attr_var=with_attr_var)
            )

        return next_ops

    def get_variables_of_attr(self, attr):
        inputs = []
        # TODO: Support recursive
        for k, v in attr.to_dict().items():
            if isinstance(v, Variable):
                inputs.append(v.name)
            elif isinstance(v, (tuple, list)):
                for _vi in v:
                    if isinstance(_vi, Variable):
                        inputs.append(_vi.name)
            elif isinstance(v, dict):
                for _vi in v.values():
                    if isinstance(_vi, Variable):
                        inputs.append(_vi.name)
        return inputs

    def is_multi_ouputs_operator(self, op: Operator):
        if len(op.outputs) > 1:
            return False
        if not isinstance(op.outputs[1], str):
            return False
        out_var = self.get_variable(op.outputs[1])
        if not isinstance(out_var, Variable):
            return False

        return out_var.var_type == VariableType.MULTI_OUTPUTS

    def clear_unused_vars(self):
        vars = list(self.variables.values())
        for var in vars:
            if self.is_leaf_variable(var):
                dst_ops = self.get_dst_operators(var)
                if len(dst_ops) == 0:
                    self.delete_variable(var)

    def toposort(self) -> List[Operator]:
        num_inputs_of_operator = {}
        op_queue = []
        for op in self.operators.values():
            num_inputs = len(self.get_previous_operators(op, with_attr_var=True))
            num_inputs_of_operator[op.name] = num_inputs
            if num_inputs == 0:
                op_queue.append(op)

        sorted_ops = []
        while len(op_queue) != 0:
            op = op_queue.pop(0)
            next_ops = self.get_next_operators(op, with_attr_var=True)
            for next_op in next_ops:
                num_inputs_of_operator[next_op.name] -= 1
                if num_inputs_of_operator[next_op.name] == 0:
                    op_queue.append(next_op)

            sorted_ops.append(op)

        if len(sorted_ops) != len(self.operators):
            print(
                "Warning: Topological sorting result is incomplete "
                "due to the existence of multiple subgraphs."
            )
        return sorted_ops

    def copy(self):
        new_graph = copy.deepcopy(self)
        new_graph.inputs.clear()
        new_graph.outputs.clear()

        for input in self.input_names:
            new_graph.add_input(new_graph.get_variable(input))

        for output in self.output_names:
            new_graph.add_output(new_graph.get_variable(output))

        return new_graph

    def get_used_node_names(self):
        """
        Get used nodes and tensors from the graph.
        """
        used_tensor_names = [i.name for i in self._outputs.values()]
        used_node_names = set()

        index = 0
        while index < len(used_tensor_names):
            used_tensor_name = used_tensor_names[index]
            index += 1

            used_tensor = self.get_variable(used_tensor_name)
            node = self.get_src_operator(used_tensor)
            if node is not None and node.name not in used_node_names:
                used_node_names.add(node.name)
                for name in node.inputs:
                    used_tensor_names.append(name)

        return list(used_node_names), used_tensor_names
