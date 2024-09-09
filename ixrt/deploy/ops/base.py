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
from abc import abstractmethod
from typing import Any, List

import onnx
from ixrt.deploy.backend.onnx.converter import convert_onnx_operator
from ixrt.deploy.backend.onnx.export import (
    default_export_onnx_operator,
    export_onnx_operator,
)
from ixrt.deploy.backend.onnx.onnx_target import OnnxGraph
from ixrt.deploy.backend.torch.executor.operators import registe_executor_operator
from ixrt.deploy.ir import BaseExecutor, Graph, Operator
from ixrt.deploy.ir.operator_attr import BaseOperatorAttr
from ixrt.deploy.quantizer import QuantOperatorObserverConfig
from ixrt.deploy.quantizer.quant_operator import registe_quant_operator
from torch.fx import GraphModule
from torch.fx import Node as FxNode

__all__ = [
    "registe_operator",
    "BaseOperator",
    "GraphModule",
    "FxNode",
    "OnnxGraph",
    "Graph",
    "Operator",
    "BaseExecutor",
    "BaseOperatorAttr",
    "registe_quant_operator",
]


def _register_operator(op_type: str, operator: "BaseOperator"):
    convert_onnx_operator(op_type, operator.convert_onnx_operator)
    export_onnx_operator(op_type, operator.export_onnx_operator)
    registe_executor_operator(op_type, operator.__call__)

    if hasattr(operator.quantize, "__isabstractmethod__"):
        if operator.quantize.__isabstractmethod__:
            return
    registe_quant_operator(op_type, operator.quantize)


def registe_operator(op_type: str, operator: "BaseOperator" = None):
    if operator is not None:
        _register_operator(op_type, operator)
        return

    def _wrap(operator_cls):
        if inspect.isclass(operator_cls):
            _operator = operator_cls()
        else:
            _operator = operator_cls
        _register_operator(op_type, _operator)
        return operator_cls

    return _wrap


class BaseOperator(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        raise NotImplementedError()

    def convert_fx_operator(
        self, ir_graph: Graph, fx_graph: GraphModule, node: FxNode
    ) -> Operator:
        raise NotImplementedError()

    def export_onnx_operator(
        self,
        ir_graph: Graph,
        operator: Operator,
        opset_imports,
        onnx_graph: OnnxGraph,
        *args,
        **kwargs,
    ) -> onnx.NodeProto:
        return default_export_onnx_operator(
            ir_graph, operator, opset_imports, onnx_graph, *args, **kwargs
        )

    @abstractmethod
    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        pass

    def call(
        self,
        executor: BaseExecutor,
        operator: Operator,
        inputs: List,
        attr: BaseOperatorAttr,
    ) -> Any:
        raise NotImplementedError()
