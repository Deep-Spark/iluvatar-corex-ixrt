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

import os.path
from typing import Iterable, Mapping

import numpy as np
import onnx
import torch
from onnx import helper
from ixrt.deploy.ir import (
    BaseTarget,
    DataType,
    Graph,
    Operator,
    Variable,
    VariableType,
)
from ixrt.deploy.ir.data_type_mapping import (
    get_dtype_from_tensor,
    py_to_onnx_dtype,
    torch_to_ir_dtype,
)

from ...core import Registry
from .export import IR_TO_ONNX_REGISTRY, default_export_onnx_operator


class OnnxGraph(object):
    def __init__(self):
        self.inputs: list = []
        self.outputs = []
        self.initilizers = []
        self.nodes = []
        self.value_info = []


class OnnxTarget(BaseTarget):
    def __init__(
        self,
        saved_path="",
        example_inputs=None,
        name: str = None,
        ir_version: int = 8,
        opset_imports: list = None,
        save_onnx_kwargs: dict = None,
        operator_registry: Registry = None,
    ):
        super(OnnxTarget, self).__init__()
        self.saved_path = self._check_saved_path(saved_path)
        self.example_inputs = example_inputs
        self.name = "IxQuantModel" if name is None else name
        self.ir_version = ir_version
        self.save_onnx_kwargs = {} if save_onnx_kwargs is None else save_onnx_kwargs

        if isinstance(opset_imports, int):
            opset_imports = [opset_imports]

        if opset_imports is None or len(opset_imports) == 0:
            opset_imports = [11]

        if isinstance(opset_imports[0], int):
            opset_imports = [
                helper.make_opsetid(domain="", version=opset) for opset in opset_imports
            ]
        self.opset_imports = opset_imports
        self.operator_registry = operator_registry or IR_TO_ONNX_REGISTRY

    def _check_saved_path(self, path):
        if not path:
            return path
        dir = os.path.dirname(path)
        if not os.path.exists(dir) and dir not in ["", ".", "./"]:
            os.makedirs(dir, exist_ok=True)
        return path

    def export(self, graph: Graph):
        onnx_graph = OnnxGraph()

        inputs = onnx_graph.inputs
        outputs = onnx_graph.outputs
        initilizers = onnx_graph.initilizers
        nodes = onnx_graph.nodes
        value_info = onnx_graph.value_info

        for var in graph.variables.values():
            if var.name in graph.inputs:
                tensor = self.build_tensor(var, with_value=False)
                inputs.append(tensor)

            elif var.name in graph.outputs:
                flatten_outs = self.flatten_outputs(var)
                for _var in flatten_outs:
                    tensor = self.build_tensor(_var, with_value=False)
                    outputs.append(tensor)

            elif graph.is_leaf_variable(var) and var.value is not None:
                tensor = self.build_tensor(var, with_value=True)
                initilizers.append(tensor)

            tensor_info = self.make_tensor_value_info(var)
            value_info.append(tensor_info)

        for operator in graph.operators.values():
            node = self.build_node(graph, operator, onnx_graph)
            nodes.append(node)

        onnx_graph = helper.make_graph(
            name=self.name,
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializer=initilizers,
            value_info=value_info,
        )

        onnx_model = helper.make_model(
            onnx_graph, producer_name="ixquant", opset_imports=self.opset_imports
        )
        onnx_model.ir_version = self.ir_version

        if self.saved_path:
            onnx.save(onnx_model, self.saved_path, **self.save_onnx_kwargs)

        return onnx_model

    def build_tensor(self, variable: Variable, with_value):
        if with_value:
            var_value = variable.value
            if var_value is None:
                raise RuntimeError(f"The value of parameter `{variable.name}` is none.")

            if not torch.is_tensor(var_value):
                var_value = torch.tensor(var_value)

            raw, is_raw_format = self.encode_tensor(var_value)
            return helper.make_tensor(
                name=variable.name,
                data_type=torch_to_ir_dtype(var_value.dtype),
                dims=tuple(var_value.shape),
                vals=raw,
                raw=is_raw_format,
            )

        elif torch.is_tensor(variable.value):
            return helper.make_tensor_value_info(
                name=variable.name,
                elem_type=torch_to_ir_dtype(variable.value.dtype),
                shape=variable.value.shape,
            )

        elif isinstance(variable.value, np.ndarray):
            return helper.make_tensor_value_info(
                name=variable.name, elem_type=variable.dtype, shape=variable.value.shape
            )

        elif isinstance(variable.value, (tuple, list)):
            dtype = float

            def find_dtype(v):
                nonlocal dtype
                if isinstance(v, Mapping):
                    if len(v) > 0:
                        find_dtype(list(v.values())[0])
                elif isinstance(v, (tuple, list)):
                    if len(v) > 0:
                        find_dtype(v[0])
                elif torch.is_tensor(v):
                    dtype = torch_to_ir_dtype(v.dtype)
                elif isinstance(v, np.ndarray):
                    dtype = torch_to_ir_dtype(torch.from_numpy(v).dtype)
                else:
                    dtype = type(v)

            find_dtype(variable.value)

            onnx_dtype = py_to_onnx_dtype(dtype)

            return helper.make_sequence(
                variable.name, elem_type=onnx_dtype, values=variable.value
            )

        elif isinstance(variable.value, Mapping):
            key_type = py_to_onnx_dtype(list(variable.value.keys())[0])
            return helper.make_map(
                name=variable.name,
                key_type=key_type,
                keys=variable.value.keys(),
                values=variable.value.values(),
            )

        if variable.dtype in [DataType.UNDEFINED, None]:
            dtype = py_to_onnx_dtype(type(variable.value))
        else:
            dtype = variable.dtype.value

        return helper.make_tensor_value_info(
            name=variable.name, elem_type=dtype, shape=variable.shape
        )

    def make_tensor_value_info(self, variable):
        dtype = variable.dtype
        shape = variable.shape

        if variable.dtype not in [DataType.UNDEFINED, None]:
            dtype = variable.dtype
        elif variable.value is not None:
            if torch.is_tensor(variable.value) or isinstance(
                variable.value, np.ndarray
            ):
                dtype = get_dtype_from_tensor(variable.value)
            else:
                dtype = py_to_onnx_dtype(type(variable.value))

        if variable.shape is None:
            if torch.is_tensor(variable.value) or isinstance(
                variable.value, np.ndarray
            ):
                shape = list(variable.value.shape)

        return helper.make_tensor_value_info(
            name=variable.name, elem_type=dtype, shape=shape
        )

    def build_node(self, graph: Graph, operator: Operator, onnx_graph):
        exporter = self.operator_registry.get(operator.op_type, default=None)
        if exporter is None:
            exporter = default_export_onnx_operator
        node = exporter(graph, operator, self.opset_imports, onnx_graph=onnx_graph)
        return node

    def encode_tensor(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor, False

        if tensor.numel() == 0:
            return [], False

        if tensor.ndim == 0:
            return [tensor.detach().cpu().item()], False

        if tensor.ndim >= 1:
            tensor_value = tensor.detach().cpu().numpy()
            return tensor_value.tobytes(), True

        return tensor, False

    def flatten_outputs(self, var: Variable):
        outs = []

        def flatten_var(_var):
            if isinstance(_var, Variable):
                if _var.var_type == VariableType.LIST:
                    for _var_item in _var.value:
                        flatten_var(_var_item)
                else:
                    outs.append(_var)

            elif isinstance(_var, (list, tuple)):
                for _var_item in _var:
                    flatten_var(_var_item)

            elif isinstance(_var, Mapping):
                for _var_item in _var.values():
                    flatten_var(_var_item)

            else:
                outs.append(_var)

        flatten_var(var)
        return outs
