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

import torch
from ixrt.deploy.fusion import BasePass
from ixrt.deploy.ir.data_type import DataType
from ixrt.deploy.ir.graph import Graph
from ixrt.deploy.ir.graph_transform.transform import GraphTransform
from ixrt.deploy.ir.operator_attr import *

from ..quant_function import quantize


class AddQdqPair(BasePass):
    def __init__(self):
        self.transformer = None

    def process(self, graph: Graph) -> Graph:
        self.transformer = GraphTransform(graph=graph)
        ops = self.transformer.operators
        last_op = list(ops.keys())[-1]
        for op in list(ops.values()):
            if op.is_quant_operator:
                self.add_qdq_pair(op, last_op)
            else:
                self.add_quant(op)

        return self.transformer.graph

    def add_qdq_pair(self, operator, last_op_name):
        deq_output = []

        for var_name in operator.inputs:
            if self.transformer.is_quant_variable(var_name):
                qparam = self.transformer.get_quant_parameter(var_name)
                scale_var = self.transformer.make_variable(
                    name=f"{var_name}_scale", value=qparam.scale
                )
                zp_var = self.transformer.make_variable(
                    name=f"{var_name}_zero_point",
                    value=torch.tensor(qparam.zero_point).to(dtype=torch.int8),
                )
                deq_output_var = self.transformer.make_variable(
                    name=f"{var_name}_DequantizeLinear_Output", dtype=DataType.FLOAT
                )
                attrs = dict(axis=qparam.quant_dim) if qparam.per_channel else dict()

                if self.transformer.get_src_operator(var_name) is not None and (
                    not self.transformer.get_src_operator(var_name).is_quant_operator
                ):
                    cast_output_var = self.transformer.make_variable(
                        name=f"{var_name}_cast_Output", dtype=DataType.INT8
                    )
                    cast_op = self.transformer.make_operator(
                        name=f"{var_name}__Cast",
                        op_type="Cast",
                        inputs=[var_name],
                        outputs=[cast_output_var.name],
                        attr_cls=CastAttr,
                        to=DataType.INT8,
                    )
                    dequant_inputs = [cast_output_var.name, scale_var.name, zp_var.name]

                    dequantize_op = self.transformer.make_operator(
                        name=f"{var_name}__DequantizeLinear",
                        op_type="DequantizeLinear",
                        inputs=dequant_inputs,
                        outputs=[deq_output_var.name],
                        attr_cls=DequantizeLinearAttr if qparam.per_channel else BaseOperatorAttr,
                        **attrs
                    )
                    deq_output.append(deq_output_var.name)
                elif not self.transformer.is_leaf_variable(var_name) or (
                    var_name == self.transformer.graph.first_input.name
                ):
                    self.create_qdq_ops(
                        var_name,
                        var_name,
                        deq_output_var.name,
                        scale_var.name,
                        zp_var.name,
                        qparam.per_channel,
                        qparam.quant_dim
                    )
                    deq_output.append(deq_output_var.name)
                else:
                    quantized_tensor = quantize(
                        torch.tensor(
                            self.transformer.graph.get_var_value(var_name)
                        ).cpu(),
                        qparam,
                    ).to(dtype=torch.int8)
                    self.transformer.graph.set_var_value(
                        var_name, value=quantized_tensor
                    )
                    dequant_inputs = [var_name, scale_var.name, zp_var.name]
                    dequantize_op = self.transformer.make_operator(
                        name=f"{var_name}_DequantizeLinear",
                        op_type="DequantizeLinear",
                        inputs=dequant_inputs,
                        outputs=[deq_output_var.name],
                        attr_cls=DequantizeLinearAttr if qparam.per_channel else BaseOperatorAttr,
                        **attrs
                    )
                    deq_output.append(deq_output_var.name)

            else:
                deq_output.append(var_name)

        operator.replace_inputs(deq_output)

        if len(self.transformer.graph.get_next_operators(operator)) == 0:
            op_outputs = []
            for var_name in operator.outputs:
                if self.transformer.is_quant_variable(var_name):
                    qparam = self.transformer.get_quant_parameter(var_name)
                    input_var = self.transformer.make_variable(
                        name=f"{var_name}_QuantizeLinear_Input", dtype=DataType.FLOAT
                    )
                    scale_var = self.transformer.make_variable(
                        name=f"{var_name}_scale", value=qparam.scale
                    )
                    zp_var = self.transformer.make_variable(
                        name=f"{var_name}_zero_point",
                        value=torch.tensor(qparam.zero_point).to(dtype=torch.int8),
                    )
                    op_outputs.append(input_var.name)
                    self.create_qdq_ops(
                        var_name, input_var.name, var_name, scale_var.name, zp_var.name,
                        qparam.per_channel,
                        qparam.quant_dim
                    )
            operator.replace_outputs(op_outputs)

    def add_quant(self, operator):
        deq_output = []
        for var_name in operator.inputs:
            if self.transformer.is_quant_variable(var_name):
                if self.transformer.get_src_operator(var_name) is not None and (
                    self.transformer.get_src_operator(var_name).is_quant_operator
                ):
                    qparam = self.transformer.get_quant_parameter(var_name)
                    scale_var = self.transformer.make_variable(
                        name=f"{var_name}_scale", value=qparam.scale
                    )
                    zp_var = self.transformer.make_variable(
                        name=f"{var_name}_zero_point",
                        value=torch.tensor(qparam.zero_point).to(dtype=torch.int8),
                    )
                    deq_output_var = self.transformer.make_variable(
                        name=f"{var_name}_DequantizeLinear_Output", dtype=DataType.FLOAT
                    )
                    if not self.transformer.is_leaf_variable(var_name):
                        self.create_qdq_ops(
                            var_name,
                            var_name,
                            deq_output_var.name,
                            scale_var.name,
                            zp_var.name,
                            qparam.per_channel,
                            qparam.quant_dim
                        )
                        deq_output.append(deq_output_var.name)
                    else:
                        # Reserve the leaf variable
                        deq_output.append(var_name)
                else:
                    # Reserve the variable whose source op is unquantized but several target ops are quantized
                    deq_output.append(var_name)
            else:
                # Reserve the unquantized variable
                deq_output.append(var_name)

        operator.replace_inputs(deq_output)

    def create_qdq_ops(self, var_name, input_name, output_name, scale_name, zp_name, per_channel, axis):
        quant_inputs = [input_name, scale_name, zp_name]
        quant_output = self.transformer.make_variable(
            name=f"{var_name}_QuantizeLinear_Output", dtype=DataType.INT8
        )

        attrs = dict(axis=axis) if per_channel else dict()
        quantize_op = self.transformer.make_operator(
            name=f"{var_name}_QuantizeLinear",
            op_type="QuantizeLinear",
            inputs=quant_inputs,
            outputs=[quant_output.name],
            attr_cls=QuantizeLinearAttr if per_channel else BaseOperatorAttr,
            **attrs
        )

        dequant_inputs = [quant_output.name, scale_name, zp_name]

        dequantize_op = self.transformer.make_operator(
            name=f"{var_name}_DequantizeLinear",
            op_type="DequantizeLinear",
            inputs=dequant_inputs,
            outputs=[output_name],
            attr_cls=DequantizeLinearAttr if per_channel else BaseOperatorAttr,
            **attrs
        )
