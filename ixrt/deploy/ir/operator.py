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
from typing import Dict, List

from .operator_attr import BaseOperatorAttr, EmptyAttr


class Operator(object):
    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: BaseOperatorAttr = None,
    ):
        self._name = name
        self._op_type = op_type
        self._inputs = inputs
        self._outputs = outputs
        self._attributes = attributes if attributes is not None else EmptyAttr()
        self._is_quant_op = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    def op_type(self, new_type):
        self._op_type = new_type

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def num_outputs(self):
        return len(self.outputs)

    def replace_inputs(self, new_inputs):
        self._inputs = new_inputs

    def replace_outputs(self, new_outputs):
        self._outputs = new_outputs

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, new_attr):
        self._attributes = new_attr

    @property
    def is_quant_operator(self):
        return self._is_quant_op

    @is_quant_operator.setter
    def is_quant_operator(self, is_quant):
        self._is_quant_op = is_quant

    def mark_as_quant_op(self):
        self.is_quant_operator = True

    def unmark_as_quant_op(self):
        self.is_quant_operator = False

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return (
            f"Operator(name={self.name}, op_type={self.op_type}, "
            f"inputs={self.inputs}, outputs: {self.outputs}, "
            f"attributes={self.attributes})"
        )
