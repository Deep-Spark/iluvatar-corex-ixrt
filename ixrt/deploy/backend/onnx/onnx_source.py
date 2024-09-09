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

import onnx
from onnx import numpy_helper
from ixrt.deploy.ir import (
    BaseSource,
    DataType,
    Graph,
    Placeholder,
    Variable,
    VariableOptions,
)
from ixrt.deploy.ir import data_type_mapping as dtype_mapping
from ixrt.deploy.ir import generate_operator_name
from ixrt.deploy.quantizer.save_quant_param import SaveQuantParameterPPQStyle

from ...core.producer_type import ProducerType
from .converter import get_converter
from .operator_type_mapping import onnx2ir_op
from .quant_parameter_serializer import unpack_quant_params

__all__ = ["OnnxSource"]


class OnnxSource(BaseSource):
    def __init__(self, path: str, load_kwargs: dict = None):
        super(OnnxSource, self).__init__()
        self.path = path
        self.load_kwargs = load_kwargs or {}

    def convert_to_ir(self) -> Graph:
        ir_graph = Graph()
        ir_graph.producers.append(ProducerType.ONNX)
        onnx_graph = onnx.load(self.path, **self.load_kwargs)
        ir_graph.set_meta(onnx_graph)
        parser_seq = [
            self.add_variables,
            self.add_inputs,
            self.add_output,
            self.add_operators,
            self.add_value_info,
            self.add_quant_anno,
        ]

        for parser in parser_seq:
            parser(ir_graph, onnx_graph.graph)

        return ir_graph

    def add_inputs(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        for input in onnx_graph.input:
            if ir_graph.containe_var(input.name):
                print(f"Ignore input {input.name}, because it already exists.")
                continue
            options = self._make_var_options_from_node(input)
            var = Variable(name=input.name, options=options)
            ir_graph.add_input(var)

    def add_output(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        for output in onnx_graph.output:
            options = self._make_var_options_from_node(output)
            var = Variable(name=output.name, options=options)
            ir_graph.add_output(var)

    def add_variables(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        for tensor in onnx_graph.initializer:
            tensor: onnx.TensorProto
            options = VariableOptions(
                dtype=dtype_mapping.onnx_to_ir_dtype(tensor.data_type),
                shape=list(tensor.dims),
            )

            var = Variable(
                name=tensor.name, value=numpy_helper.to_array(tensor), options=options
            )
            ir_graph.add_variable(var)

    def add_operators(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        def add_placeholders(names):
            for name in names:
                if name not in ir_graph.variables:
                    var = Placeholder(name)
                    ir_graph.add_variable(var)

        for op in onnx_graph.node:
            op_type = onnx2ir_op(op.op_type)
            if op.name in ["", None]:
                op.name = generate_operator_name(ir_graph, pattern=op_type + "_{idx}")
            converter = get_converter(op_type)

            add_placeholders(op.input)
            add_placeholders(op.output)
            ir_op = converter(ir_graph, onnx_graph, op)
            ir_graph.add_operator(ir_op)

    def add_value_info(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        for vinfo in onnx_graph.value_info:
            var_name = vinfo.name
            if ir_graph.containe_var(var_name):
                var = ir_graph.get_variable(var_name)
                if var.shape is None:
                    var.options.shape = self._parse_onnx_shape(
                        vinfo.type.tensor_type.shape
                    )

                if var.dtype in [DataType.UNDEFINED, None]:
                    var.options.dtype = dtype_mapping.onnx_to_ir_dtype(
                        vinfo.type.tensor_type.elem_type
                    )

    def add_quant_anno(self, ir_graph: Graph, onnx_graph: onnx.GraphProto):
        if not hasattr(onnx_graph, "quantization_annotation"):
            return

        quant_params = unpack_quant_params(onnx_graph.quantization_annotation)
        SaveQuantParameterPPQStyle.load(ir_graph, quant_params=quant_params)

    def _make_var_options_from_node(self, node):
        def get_shape():
            if (
                hasattr(node, "type")
                and hasattr(node.type, "tensor_type")
                and hasattr(node.type.tensor_type, "shape")
            ):
                return self._parse_onnx_shape(node.type.tensor_type.shape)
            return None

        def get_dtype():
            if hasattr(node, "type") and hasattr(node.type, "tensor_type"):
                return dtype_mapping.onnx_to_ir_dtype(node.type.tensor_type.elem_type)
            return None

        return VariableOptions(
            var_type=dtype_mapping.onnx_to_ir_var_type(node.type),
            dtype=get_dtype(),
            shape=get_shape(),
        )

    def _parse_onnx_shape(self, shape):
        if hasattr(shape, "dim"):
            shape = shape.dim

        ir_shape = []
        for dim in shape:
            if hasattr(dim, "dim_param") and dim.dim_param not in [None, ""]:
                ir_shape.append(dim.dim_param)
            elif hasattr(dim, "dim_value"):
                ir_shape.append(dim.dim_value)

        return ir_shape
