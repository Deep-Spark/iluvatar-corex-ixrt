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

import dataclasses
from typing import List

import onnx
import torch
import torch.nn.functional as F

from ...backend.onnx.converter import convert_onnx_operator, default_converter
from ...backend.torch.executor.operators._operators import to_py_type
from ...ir import Graph, GraphTransform, Operator, Variable
from ...ir import operator_attr as attr
from ...ir import operator_attr as op_attrs
from ...ir.operator_attr import BaseOperatorAttr
from ...ir.operator_type import OperatorType as OP
from ...ops.base import *
from ...quantizer import QuantOperatorObserverConfig, QuantPolicy
from ...quantizer.quant_operator.base import quant_single_input_operator
from ..base_pass import BasePass, PassSequence, registe_pass
from ..level0 import ConstantPass, FormatClip, FormatReshape
from ..level2 import ClearUnusedVariablesPass
from ..matcher import PatternGraph


def get_constant_input_name_of_operator(graph: Graph, operator: Operator):
    const = None
    for input in operator.inputs:
        if not graph.containe_var(input):
            continue

        if not graph.is_leaf_variable(input):
            continue

        input_var = graph.get_variable(input)
        if input_var.value is not None:
            const = input
    return const


class ReshapeToFlattenPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.SHAPE, OP.SLICE, OP.CONCAT, OP.RESHAPE],
            self.reshape_to_flatten,
            strict=True,
        )
        return graph

    def reshape_to_flatten(self, graph, pattern: PatternGraph):
        shape = pattern.nodes[0]
        reshape = pattern.nodes[-1]

        reshape_op: Operator = reshape.operator
        slice_op: Operator = pattern.nodes[1].operator

        if shape.operator.inputs[0] != reshape_op.inputs[0]:
            return

        starts = to_py_type(graph.get_variable(slice_op.inputs[1]).value)
        ends = to_py_type(graph.get_variable(slice_op.inputs[2]).value)
        axes = to_py_type(graph.get_variable(slice_op.inputs[3]).value)
        if starts != [0] and axes != [0]:
            return

        if not isinstance(ends, list) or len(ends) == 0 or not isinstance(ends[0], int):
            return

        self.transform.delete_operators_between_op_op(
            shape.operator, pattern.nodes[2].operator
        )

        reshape_op.op_type = OP.FLATTEN
        reshape_op.replace_inputs([reshape_op.inputs[0]])
        reshape_op.attributes = attr.FlattenAttr(ends[0])


class FuseLayerNormPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.REDUCE_MEAN,
                OP.SUB,
                OP.POW,
                OP.REDUCE_MEAN,
                OP.ADD,
                OP.SQRT,
                OP.DIV,
                OP.MUL,
                OP.ADD,
            ],
            self.fuse_layer_norm,
            strict=False,
        )
        return graph

    def fuse_layer_norm(self, graph: Graph, pattern: PatternGraph):
        # 检查 REDUCE_MEAN 的输入是否和 SUB 的输入是一致的
        if pattern.nodes[0].operator.inputs[0] != pattern.nodes[1].operator.inputs[0]:
            return

        # 检查 POW 的输入是否和 DIV 的输入是一致的
        if pattern.nodes[2].operator.inputs[0] != pattern.nodes[6].operator.inputs[0]:
            return

        # 检查部分算子的输出是否被多个算子使用
        nodes = pattern.nodes
        for node in [nodes[0]] + nodes[2:-1]:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) > 1:
                return

        eps = None
        for input in nodes[4].operator.inputs:
            input_var = graph.get_variable(input)
            if input_var.value is not None and graph.is_leaf_variable(input):
                eps = to_py_type(input_var.value)

        scale = get_constant_input_name_of_operator(graph, nodes[-2].operator)
        bias = get_constant_input_name_of_operator(graph, nodes[-1].operator)

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        layer_norm_op = self.transform.make_operator(
            op_type=OP.LAYER_NORM,
            inputs=[nodes[0].operator.inputs[0], scale, bias],
            outputs=[nodes[-1].operator.outputs[0]],
            axis=nodes[0].operator.attributes.axes,
            epsilon=eps,
            attr_cls=op_attrs.LayerNormAttr,
        )
        self.transform.add_operator(layer_norm_op)


class FuseGemmPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        self.transform.find_sequence_subgraph(
            pattern=[OP.MATMUL, OP.ADD], callback=self.fuse_gemm, strict=True
        )
        return graph

    def fuse_gemm(self, graph, pattern: PatternGraph):
        matmul = pattern.nodes[0]
        add = pattern.nodes[1]

        if len(add.operator.inputs) != 2:
            return

        b_var = graph.get_variable(matmul.operator.inputs[1])
        if not graph.is_leaf_variable(b_var) or b_var.value is None:
            return

        if b_var.value.ndim != 2:
            return

        bias_var = None
        for input in add.operator.inputs:
            if input not in matmul.operator.outputs:
                bias_var = input

        matmul.operator.inputs.append(bias_var)
        self.transform.delete_operator_and_link(
            add.operator, link_input=matmul.operator.outputs[0]
        )

        matmul.operator.op_type = OP.GEMM
        matmul.operator.attributes = attr.GemmAttr(transB=1)
        b_var.value = b_var.value.transpose(1, 0)


class FuseNormalizePass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        self.transform.find_sequence_subgraph(
            pattern=[OP.REDUCE_L2, OP.CLIP, OP.EXPAND, OP.DIV],
            callback=self.fuse_normlize,
            strict=True,
        )
        return graph

    def fuse_normlize(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        if nodes[0].operator.inputs[0] != nodes[-1].operator.inputs[0]:
            return

        eps = get_constant_input_name_of_operator(graph, nodes[1].operator)
        if eps is None:
            eps = 1e-12
        else:
            eps = to_py_type(graph.get_variable(eps).value)

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        norm_op = self.transform.make_operator(
            op_type="TorchNormalize",
            inputs=[nodes[0].operator.inputs[0]],
            outputs=[nodes[-1].operator.outputs[0]],
            p=2.0,
            axis=nodes[0].operator.attributes.axes,
            epsilon=eps,
        )
        self.transform.add_operator(norm_op)

        # 删除 Shape 算子
        prev_op = self.transform.get_previous_operators(norm_op)[0]
        next_ops = self.transform.get_next_operators(prev_op)
        if norm_op not in next_ops or len(next_ops) != 2:
            return

        next_ops.remove(norm_op)
        if next_ops[0].op_type == OP.SHAPE:
            self.transform.delete_operator(next_ops[0])


class FuseGeluPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        self.transform.find_sequence_subgraph(
            pattern=[OP.DIV, OP.ERF, OP.ADD, OP.MUL, OP.MUL],
            callback=self.fuse_gelu,
            strict=True,
        )
        return graph

    def fuse_gelu(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        prev_op = self.transform.get_previous_operators(nodes[0].operator)[0]
        next_ops = self.transform.get_next_operators(prev_op)
        if len(next_ops) != 2:
            return

        if nodes[0].operator not in next_ops or nodes[3].operator not in next_ops:
            return

        gelu_op_input = None
        for input in nodes[3].operator.inputs:
            if input in nodes[0].operator.inputs:
                gelu_op_input = input
                break

        self.transform.delete_operators_between_op_op(
            nodes[0].operator, nodes[-1].operator
        )

        gelu_op = self.transform.make_operator(
            op_type=OP.GELU,
            inputs=[gelu_op_input],
            outputs=[nodes[-1].operator.outputs[0]],
        )
        self.transform.add_operator(gelu_op)


@dataclasses.dataclass
class NormalizeAttr(BaseOperatorAttr):
    p: float = 2.0
    epsilon: float = 1e-12
    axis: int = 1


@registe_operator("TorchNormalize")
class NormalizeOperator(BaseOperator):
    def call(
        self,
        executor,
        operator: Operator,
        inputs: List,
        attr: NormalizeAttr,
    ):
        return F.normalize(inputs[0], p=attr.p, eps=attr.epsilon, dim=attr.axis)

    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        return default_converter(ir_graph, onnx_graph, node, attr_cls=NormalizeAttr)

    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        return quant_single_input_operator(
            graph, op, operator_observer_config, quant_outputs=quant_outputs
        )


@registe_operator(OP.GELU)
class GeluOperator(BaseOperator):
    def call(
        self,
        executor,
        operator: Operator,
        inputs: List,
        attr: NormalizeAttr,
    ):
        return F.gelu(inputs[0])

    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        return default_converter(ir_graph, onnx_graph, node, attr_cls=attr.EmptyAttr)

    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        return quant_single_input_operator(
            graph, op, operator_observer_config, quant_outputs=quant_outputs
        )


def get_quant_scale(graph: Graph, name):
    qparam = graph.get_quant_parameter(name)
    return qparam.scale


@dataclasses.dataclass
class LayerNormResidualAttr(attr.LayerNormAttr):
    residual_scale: float = None


@convert_onnx_operator("LayerNormalizationResidual")
def convert_ln_res(ir_graph: Graph, onnx_graph: onnx.GraphProto, node, attr_cls=None):
    return default_converter(ir_graph, onnx_graph, node, LayerNormResidualAttr)


class FuseLayerNormResidualPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        ops = list(graph.operators.values())
        for op in ops:
            if op.op_type != OP.ADD:
                continue

            prev_ops = graph.get_previous_operators(op)
            prev_ops = [_op.op_type for _op in prev_ops]
            if OP.LAYER_NORM in prev_ops:
                self.fuse_layer_norm(graph, op)

        return graph

    def fuse_layer_norm(self, graph: Graph, op: Operator):
        layer_norm = graph.get_src_operator(op.inputs[1])
        if layer_norm.op_type != OP.LAYER_NORM:
            return

        add = op

        scale = graph.get_quant_parameter(add.inputs[0]).scale

        layer_norm.inputs.append(add.inputs[0])
        ln_attr = layer_norm.attributes
        layer_norm.attributes = LayerNormResidualAttr(
            axis=ln_attr.axis,
            epsilon=ln_attr.epsilon,
            stash_type=ln_attr.stash_type,
            residual_scale=to_py_type(scale),
        )

        layer_norm.op_type = "LayerNormalizationResidual"
        layer_norm.outputs[0] = add.outputs[0]
        self.transform.delete_operator(add)


class ShareQuantParams(BasePass):
    def __init__(self, op_types):
        self.op_types = op_types

    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if op.op_type in self.op_types:
                self.share_quant_params(graph, op)
        return graph

    def share_quant_params(self, graph, operator):
        input_quant_params = graph.get_quant_parameter(operator.inputs[0])
        graph.add_quant_parameter(operator.outputs[0], input_quant_params)


@dataclasses.dataclass
class FuseLnGpAttr(attr.LayerNormAttr):
    ln_scale: float = None


class FuseLayerNormGlobalPoolPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            pattern=[OP.LAYER_NORM, OP.TRANSPOSE, OP.GLOBAL_AVG_POOL, OP.FLATTEN],
            callback=self.fuse_layer_norm_global_pool,
            strict=True,
        )
        return graph

    def fuse_layer_norm_global_pool(self, graph, pattern: PatternGraph):
        self.transform.delete_operators_between_op_op(
            pattern.nodes[1].operator, pattern.nodes[-1].operator
        )
        ln_op: Operator = pattern.nodes[0].operator

        ln_scale = graph.get_quant_parameter(ln_op.outputs[0])

        ln_attr: LayerNormResidualAttr = ln_op.attributes
        ln_op.attributes = FuseLnGpAttr(
            epsilon=ln_attr.epsilon, axis=ln_attr.axis, ln_scale=ln_scale.scale
        )
        ln_op.op_type = "FusedLayerNormGlobalPool"
        ln_op.replace_outputs([pattern.nodes[-1].operator.outputs[0]])


class NHWC2NCHW(BasePass):
    def process(self, graph: Graph) -> Graph:
        transform = GraphTransform(graph)

        conv = graph.get_operator("Conv_3")
        flatten = graph.get_operator("Reshape_11")
        transpose = graph.get_operator("Transpose_12")
        conv.replace_outputs([transpose.outputs[0]])
        graph.delete_operator(transpose)
        graph.delete_operator(flatten)
        return graph


class FormatLayerNorm(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if "LayerNorm" in op.op_type:
                self.format_layer_norm(graph, op)
        return graph

    def format_layer_norm(self, graph, operator):
        if not hasattr(operator.attributes, "axis"):
            return
        if isinstance(operator.attributes.axis, (tuple, list)):
            operator.attributes.axis = operator.attributes.axis[0]


class FormatTorchNormalize(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if (
                "Torch::Normalize" in op.op_type
                or "Torch_Normalize" in op.op_type
                or "TorchNormalize" in op.op_type
            ):
                self.format_torch_norm(graph, op)
        return graph

    def format_torch_norm(self, graph, operator):
        if not hasattr(operator.attributes, "axis"):
            return
        if isinstance(operator.attributes.axis, (tuple, list)):
            operator.attributes.axis = operator.attributes.axis[0]


class FuseMulAdd(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.MATMUL, OP.MUL, OP.ADD], self.fuse_mul_add, strict=True
        )

        return graph

    def fuse_mul_add(self, graph, pattern: PatternGraph):
        mul: Operator = pattern.nodes[1].operator
        add = pattern.nodes[2].operator

        mul.replace_outputs([add.outputs[0]])
        mul.replace_inputs([mul.inputs[0], mul.inputs[1], add.inputs[1]])
        mul.op_type = "FusedElementwiseMulAdd"

        graph.delete_operator(add)


class FuseMulAddAdd(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            ["FusedElementwiseMulAdd", OP.RESHAPE, OP.ADD, OP.RESHAPE],
            self.fuse_mul_add_add,
            strict=True,
        )

        return graph

    def fuse_mul_add_add(self, graph, pattern: PatternGraph):
        mul_add: Operator = pattern.nodes[0].operator
        add: Operator = pattern.nodes[2].operator

        const1: Variable = graph.get_variable(mul_add.inputs[2])
        const2: Variable = graph.get_variable(add.inputs[1])

        const1_val = const1.value.reshape(
            1,
            const1.value.shape[0],
            const1.value.shape[1],
            const1.value.shape[2],
            const2.value.shape[3],
        )
        new_const = const1_val + const2.value

        const1.value = new_const
        const1_qparam = graph.get_quant_parameter(const1.name)
        import ixrt.deploy.quantizer.quant_function as QF

        tensor = torch.tensor(new_const)
        const1_qparam.scale = QF.compute_scale_zero_point(
            tensor.min(), tensor.max(), QuantPolicy("per_tensor")
        )[0].item()

        mul_add.replace_outputs([pattern.nodes[-1].operator.outputs[0]])

        self.transform.delete_operators_between_op_op(
            pattern.nodes[1].operator, pattern.nodes[-1].operator
        )


class FuseGemmGelu(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.GEMM, OP.GELU], self.fuse_gemm_gelu, strict=True
        )
        return graph

    def fuse_gemm_gelu(self, graph, pattern: PatternGraph):
        gemm: Operator = pattern.nodes[0].operator
        gelu: Operator = pattern.nodes[1].operator

        gemm.attributes.activation = "Gelu"
        gemm.replace_outputs([gelu.outputs[0]])

        self.transform.delete_op_with_inputs(gelu)

        print("FuseGemmBiasGelu Attributes:", gemm.attributes.to_dict())


class FuseWindowPartition(BasePass):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = self.input_shape[0]

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.RESHAPE,
                OP.SLICE,
                OP.CONCAT,
                OP.SLICE,
                OP.CONCAT,
                OP.RESHAPE,
                OP.TRANSPOSE,
                OP.RESHAPE,
                OP.RESHAPE,
            ],
            self.fuse_windows_partition,
            strict=False,
        )
        return graph

    def fuse_windows_partition(self, graph, pattern: PatternGraph):
        batch_size = graph.get_variable(graph.input_names[0]).shape
        if batch_size is None:
            batch_size = self.batch_size
            print(f"Warning: Swin-Transformer using {batch_size} as batch size.")
        else:
            batch_size = batch_size[0]

        self.transform.delete_operators_between_op_op(
            pattern.nodes[0].operator, pattern.nodes[-1].operator
        )

        slice_op = pattern.nodes[1].operator
        slice_start = to_py_type(graph.get_variable(slice_op.inputs[1]).value)
        if slice_start[0] == 0:
            shift = to_py_type(graph.get_variable(slice_op.inputs[2]))[0]
        else:
            shift = slice_start[0]

        window_size = to_py_type(
            graph.get_variable(pattern.nodes[-2].operator.inputs[1]).value
        )[1]

        print(
            f"WindowPartition shift: {shift}, windows size: {window_size}, batch: {batch_size}"
        )

        op = self.transform.make_operator(
            op_type="WindowPartition",
            inputs=[pattern.nodes[0].operator.inputs[0]],
            outputs=[pattern.nodes[-1].operator.outputs[0]],
            shift=shift,
            window_size=window_size,
            batch=batch_size,
        )

        self.transform.add_operator(op)


class FuseWindowReverse(BasePass):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = self.input_shape[0]

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.RESHAPE,
                OP.RESHAPE,
                OP.TRANSPOSE,
                OP.RESHAPE,
                OP.SLICE,
                OP.CONCAT,
                OP.SLICE,
                OP.CONCAT,
                OP.RESHAPE,
            ],
            self.fuse_windows_reverse,
            strict=False,
        )
        return graph

    def fuse_windows_reverse(self, graph, pattern: PatternGraph):
        batch_size = graph.get_variable(graph.input_names[0]).shape
        if batch_size is None:
            batch_size = self.batch_size
            print(f"Warning: Swin-Transformer using {batch_size} as batch size.")
        else:
            batch_size = batch_size[0]

        self.transform.delete_operators_between_op_op(
            pattern.nodes[0].operator, pattern.nodes[-1].operator
        )

        slice_op2 = pattern.nodes[6].operator
        slice_start = to_py_type(graph.get_variable(slice_op2.inputs[1]).value)[0]

        if slice_start == 0:
            shift = -to_py_type(graph.get_variable(slice_op2.inputs[2]).value)[0]
        else:
            shift = -slice_start

        window_size = to_py_type(
            graph.get_variable(pattern.nodes[0].operator.inputs[1]).value
        )[1]

        print(f"WindowReverse shift: {shift}, windows size: {window_size}")

        op = self.transform.make_operator(
            op_type="WindowReverse",
            inputs=[pattern.nodes[0].operator.inputs[0]],
            outputs=[pattern.nodes[-1].operator.outputs[0]],
            shift=shift,
            window_size=window_size,
            batch=batch_size,
        )

        self.transform.add_operator(op)


class NormalizeAndSplitQKV(BasePass):
    def __init__(self, enable=True):
        super(NormalizeAndSplitQKV, self).__init__()
        self.enable = enable

    def process(self, graph: Graph) -> Graph:
        if not self.enable:
            return graph

        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.TRANSPOSE, OP.GATHER, "TorchNormalize"],
            self.fuse_normalize_split_qkv,
            strict=False,
        )
        return graph

    def fuse_normalize_split_qkv(self, graph, pattern: PatternGraph):
        transpose = pattern.nodes[0].operator
        qkv_ops = [None] * 3

        delete_ops = [transpose]

        for gather_op in self.transform.get_next_operators(transpose):
            if gather_op.op_type != OP.GATHER:
                raise RuntimeError("Subgraph is not merged.")

            idx = to_py_type(graph.get_variable(gather_op.inputs[1]).value)
            next_op = self.transform.get_next_operators(gather_op)[0]
            if next_op.op_type == "TorchNormalize":
                qkv_ops[idx] = next_op
                delete_ops.extend([gather_op, next_op])
            else:
                qkv_ops[-1] = gather_op
                delete_ops.append(gather_op)

        op = self.transform.make_operator(
            "NormalizeAndSplitQKV",
            inputs=[transpose.inputs[0]],
            outputs=[qkv.outputs[0] for qkv in qkv_ops],
        )
        self.transform.add_operator(op)

        # 设置 V 的量化参数 和 Transpose 的输出保持一致
        input_quant_params = graph.get_quant_parameter(transpose.outputs[0])
        graph.add_quant_parameter(qkv_ops[-1].outputs[0], input_quant_params)

        for op in delete_ops:
            self.transform.delete_operator(op)


class FuseWindowAttention(BasePass):
    def __init__(self, enable=True):
        super(FuseWindowAttention, self).__init__()
        self.enable = enable

    def process(self, graph: Graph) -> Graph:
        if not self.enable:
            return graph

        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.MATMUL,
                "FusedElementwiseMulAdd",
                OP.SOFTMAX,
                OP.MATMUL,
                OP.TRANSPOSE,
                OP.RESHAPE,
            ],
            self.fuse_window_attention,
            strict=False,
        )
        return graph

    def fuse_window_attention(self, graph, pattern: PatternGraph):
        gemm1 = pattern.nodes[0].operator
        linear = pattern.nodes[1].operator
        softmax = pattern.nodes[2].operator
        gemm2 = pattern.nodes[3].operator
        reshape = pattern.nodes[-1].operator

        gemm1_transpose = graph.get_previous_operators(gemm1)
        if len(gemm1_transpose) != 2:
            print("Not found transpose in front of Gemm1.")
            return

        if gemm1_transpose[0].op_type == OP.TRANSPOSE:
            gemm1_transpose = gemm1_transpose[0]
        else:
            gemm1_transpose = gemm1_transpose[1]

        window_attn = self.transform.make_operator(
            "WindowAttention",
            inputs=[
                # QKV
                gemm1.inputs[0],
                gemm1_transpose.inputs[0],
                gemm2.inputs[1],
                # alpha and beta
                linear.inputs[1],
                linear.inputs[2],
            ],
            outputs=[reshape.outputs[0]],
            softmax_out_scale=get_quant_scale(graph, softmax.outputs[0]),
        )

        self.transform.add_operator(window_attn)

        self.transform.delete_op_with_outputs(gemm1_transpose)
        self.transform.delete_operators_between_op_op(gemm1, reshape)


class FusePatchMerging(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [OP.RESHAPE, OP.SLICE, OP.SLICE, OP.CONCAT],
            self.fuse_patch_merging,
            strict=False,
        )
        return graph

    def fuse_patch_merging(self, graph, pattern: PatternGraph):
        op = self.transform.make_operator(
            "PatchMerging",
            inputs=[pattern.nodes[0].operator.outputs[0]],
            outputs=[pattern.nodes[-1].operator.outputs[0]],
        )
        # 需要暂时删除掉，后面会将中间算子删除，提前添加会被删除掉
        self.transform.delete_operator(op)

        input_quant_params = graph.get_quant_parameter(op.inputs[0])
        graph.add_quant_parameter(op.outputs[0], input_quant_params)

        self.transform.delete_operators_between_var_op(
            pattern.nodes[0].operator.outputs[0], pattern.nodes[-1].operator
        )
        self.transform.add_operator(op)


class FuseAddAdd(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            ["SwinTElementwise", OP.RESHAPE, "SwinTElementwise", OP.RESHAPE],
            self.fuse_mul_add_add,
            strict=True,
        )
        return graph

    def fuse_mul_add_add(self, graph, pattern: PatternGraph):
        add1: Operator = pattern.nodes[0].operator
        add2: Operator = pattern.nodes[2].operator

        if add1.attributes.operator != "Add" or add2.attributes.operator != "Add":
            return

        const1: Variable = graph.get_variable(add1.inputs[1])
        const2: Variable = graph.get_variable(add2.inputs[1])

        s = const1.value.shape
        const1.value = const1.value.reshape(1, s[0], s[1], s[2], s[3])

        new_const = const1.value + const2.value
        new_const = new_const.reshape(1, -1, new_const.shape[-2], new_const.shape[-1])

        const1.value = new_const
        const1_qparam = graph.get_quant_parameter(const1.name)
        import ixrt.deploy.quantizer.quant_function as QF

        tensor = torch.tensor(new_const)
        const1_qparam.scale = QF.compute_scale_zero_point(
            tensor.min(), tensor.max(), QuantPolicy("per_tensor")
        )[0].item()

        add1.replace_outputs([pattern.nodes[-1].operator.outputs[0]])

        self.transform.delete_operators_between_op_op(
            pattern.nodes[1].operator, pattern.nodes[-1].operator
        )


class FuseWindowAttentionV1(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        self.transform.find_sequence_subgraph(
            [
                OP.TRANSPOSE,
                OP.GATHER,
                "SwinTElementwise",
                "SwinTMatmul",
                "SwinTElementwise",
                "SwinTSoftmax",
                "SwinTMatmul",
                "Transpose",
                OP.RESHAPE,
            ],
            self.fuse_window_attention,
            strict=False,
        )
        return graph

    def fuse_window_attention(self, graph, pattern: PatternGraph):
        softmax = pattern.nodes[5].operator
        alpha = graph.get_variable(pattern.nodes[2].operator.inputs[1])
        beta = graph.get_variable(pattern.nodes[4].operator.inputs[1])

        self.transform.delete_operators_between_op_op(
            pattern.nodes[0].operator, pattern.nodes[-1].operator
        )

        window_attn = self.transform.make_operator(
            "WindowAttentionV1",
            inputs=[pattern.nodes[0].operator.inputs[0], alpha.name, beta.name],
            outputs=[pattern.nodes[-1].operator.outputs[0]],
            softmax_out_scale=get_quant_scale(graph, softmax.outputs[0]),
        )
        self.transform.add_operator(window_attn)
        print(
            "WindowAttention Softmax Scale:", window_attn.attributes.softmax_out_scale
        )


class FormatScalar(BasePass):
    def process(self, graph: Graph):
        for var in graph.variables.values():
            var: Variable
            use_ops = graph.get_dst_operators(var)

            if len(use_ops) == 0 or (
                len(use_ops) > 1 and use_ops[0].op_type != OP.GATHER
            ):
                continue

            if use_ops[0].op_type not in [OP.MUL, OP.ADD, OP.GATHER]:
                continue

            if var.value is not None and var.value.ndim == 0:
                var.value = var.value.reshape(1)
                print(f"Reshape scalar to tensor for {var.name}.")

        return graph


class RenameSwinTOps(BasePass):
    def process(self, graph: Graph) -> Graph:
        type_mapping = {
            OP.GEMM: "SwinTGemm",
            OP.MATMUL: "SwinTMatmul",
            OP.LAYER_NORM: "SwinTLayerNormalization",
            "LayerNormalizationResidual": "SwinTLayerNormalizationResidual",
            OP.SOFTMAX: "SwinTSoftmax",
        }
        for op in graph.operators.values():
            if op.op_type in type_mapping:
                op.op_type = type_mapping[op.op_type]
                # print(f"Rename op type to {op.op_type}")

        return graph


class FullInt8Inference(BasePass):
    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            op.mark_as_quant_op()

        return graph


@registe_pass(level=2)
class SwinTransformerQuantizationPass(PassSequence):
    def __init__(self):
        super().__init__(
            ConstantPass(),
            FormatClip(),
            FuseGemmPass(),
            ReshapeToFlattenPass(),
            FuseLayerNormPass(),
            FuseNormalizePass(),
            FuseGeluPass(),
        )


SHARED_QUANT_PARAMS_OPS = [OP.GATHER, OP.SLICE, "WindowPartition", "WindowReverse"]


@registe_pass(level=2)
class SwinTransformerV1InferencePass(PassSequence):
    def __init__(self, input_shape: List):
        super().__init__(
            FormatScalar(),
            FuseLayerNormResidualPass(),
            FuseLayerNormGlobalPoolPass(),
            NHWC2NCHW(),
            FormatLayerNorm(),
            FormatTorchNormalize(),
            FormatReshape(),
            FuseWindowPartition(input_shape),
            FuseWindowReverse(input_shape),
            ShareQuantParams(SHARED_QUANT_PARAMS_OPS),
            FuseGemmGelu(),
            RenameSwinTOps(),
            FuseAddAdd(),
            # FuseWindowAttentionV1(),
            FusePatchMerging(),
            ClearUnusedVariablesPass(),
            FullInt8Inference(),
        )


@registe_pass(level=2)
class SwinTransformerV2InferencePass(PassSequence):
    def __init__(self, input_shape: List):
        super().__init__(
            FuseLayerNormResidualPass(),
            FuseLayerNormGlobalPoolPass(),
            # NHWC2NCHW(),
            FormatLayerNorm(),
            FormatTorchNormalize(),
            FormatReshape(),
            FuseWindowPartition(input_shape),
            FuseWindowReverse(input_shape),
            ShareQuantParams(SHARED_QUANT_PARAMS_OPS),
            FuseMulAdd(),
            FuseMulAddAdd(),
            FuseGemmGelu(),
            # 只针对该 Size 做融合
            NormalizeAndSplitQKV(input_shape[-1] == 256),
            FuseWindowAttention(input_shape[-1] == 256),
            FusePatchMerging(),
            RenameSwinTOps(),
            ClearUnusedVariablesPass(),
            FullInt8Inference(),
        )
