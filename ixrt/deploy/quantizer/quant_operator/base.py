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

import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from ixrt.deploy.ir import Graph, Operator
from ixrt.deploy.utils.object import get_obj_name

from ..quant_operator_config import QuantOperatorObserverConfig

QUANT_OPERATORS = dict()


def registe_quant_operator(op_type: Union[str, list], quant_op=None):
    if isinstance(op_type, str):
        op_type = [op_type]

    if quant_op is not None:
        for t in op_type:
            QUANT_OPERATORS[t] = quant_op
        return quant_op

    def wrap(fn):
        for t in op_type:
            QUANT_OPERATORS[t] = fn
        return fn

    return wrap


def get_registed_quant_operators():
    return QUANT_OPERATORS


def quant_activations(
    graph: Graph,
    op: Operator,
    operator_observer_config: QuantOperatorObserverConfig,
    num_activations: Optional[int] = 1,
    quant_outputs: bool = False,
):
    if num_activations is None:
        num_activations = len(op.inputs)

    for var in op.inputs[:num_activations]:
        activation_var = graph.get_variable(var)
        activation_var.set_value_observer(operator_observer_config.activation.copy())

    if quant_outputs:
        for var in op.outputs:
            activation_var = graph.get_variable(var)
            activation_var.set_value_observer(
                operator_observer_config.activation.copy()
            )


def quant_single_input_operator(
    graph: Graph,
    op: Operator,
    operator_observer_config: QuantOperatorObserverConfig,
    quant_outputs: bool = True,
):
    return quant_activations(
        graph,
        op,
        operator_observer_config,
        num_activations=1,
        quant_outputs=quant_outputs,
    )


def quant_double_input_operator(
    graph: Graph,
    op: Operator,
    operator_observer_config: QuantOperatorObserverConfig,
    quant_outputs: bool = True,
):
    return quant_activations(
        graph,
        op,
        operator_observer_config,
        num_activations=2,
        quant_outputs=quant_outputs,
    )


def quant_based_weight_bias_operator(
    graph: Graph,
    op: Operator,
    operator_observer_config: QuantOperatorObserverConfig,
    quant_outputs: bool = True,
):
    quant_activations(
        graph,
        op,
        operator_observer_config,
        num_activations=1,
        quant_outputs=quant_outputs,
    )

    if len(op.inputs) >= 2:
        weight = graph.get_variable(op.inputs[1])
        weight.set_value_observer(operator_observer_config.weight)

    if len(op.inputs) >= 3:
        bias = graph.get_variable(op.inputs[2])
        bias.set_value_observer(operator_observer_config.bias)
