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

from typing import Union

from ixrt.deploy.core import Registry
from ixrt.deploy.ir.operator_attr import BaseOperatorAttr

TORCH_EXECUTOR_OPERATORS = Registry("TorchExecutorOperatorRegistry")


def registe_executor_operator(op_type: Union[str, list], call_op_fn=None):
    if isinstance(op_type, str):
        op_type = [op_type]

    if call_op_fn is not None:
        for t in op_type:
            TORCH_EXECUTOR_OPERATORS.add_handler(t, call_op_fn)
        return call_op_fn

    def wrap(fn):
        for t in op_type:
            TORCH_EXECUTOR_OPERATORS.add_handler(t, fn)
        return fn

    return wrap


def get_executor_operator(op_type):
    if TORCH_EXECUTOR_OPERATORS.containe(op_type):
        return TORCH_EXECUTOR_OPERATORS.get(op_type)
    raise RuntimeError(
        f"Not found registed operator for operator `{op_type}`, "
        f"please using `registe_executor_operator` to registe the operator."
    )


def default_call_op(fn, inputs, attr: BaseOperatorAttr):
    return fn(*inputs, **attr.to_dict())
