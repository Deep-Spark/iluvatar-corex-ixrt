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

import onnx.numpy_helper
from onnx import numpy_helper
from ixrt.deploy.ir import Graph
from ixrt.deploy.ir import operator_attr as attrs
from ixrt.deploy.ir.operator_type import OperatorType as OP

from .base import convert_onnx_operator, default_converter


def mark_var_as_parameter(graph, vars):
    for var in vars:
        variable = graph.get_variable(var)
        variable.mark_as_parameter()


@convert_onnx_operator(OP.ABS)
def convert_abs(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ACOS)
def convert_acos(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ACOSH)
def convert_acosh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ADD)
def convert_add(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.AND)
def convert_and(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ARGMAX)
def convert_argmax(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ReductionAxisAttr)


@convert_onnx_operator(OP.ARGMIN)
def convert_argmin(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ReductionAxisAttr)


@convert_onnx_operator(OP.ASIN)
def convert_asin(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ASINH)
def convert_asinh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ATAN)
def convert_atan(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ATANH)
def convert_atanh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.AVG_POOL)
def convert_avg_pool(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.PoolingAttr)


@convert_onnx_operator(OP.BATCH_NORM)
def convert_batch_norm(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.BatchNormAttr)


@convert_onnx_operator(OP.CAST)
def convert_cast(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.CastAttr)


@convert_onnx_operator(OP.CAST_LIKE)
def convert_cast_like(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.CEIL)
def convert_ceil(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.CELU)
def convert_celu(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.CeluAttr)


@convert_onnx_operator(OP.CLIP)
def convert_clip(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ClipAttr)


@convert_onnx_operator(OP.CONCAT)
def convert_concat(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.CONCAT_FROM_SEQUENCE)
def convert_concat_from_sequence(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ConcatFromSequenceAttr)


@convert_onnx_operator(OP.CONV)
def convert_conv(ir_graph: Graph, *args, **kwargs):
    operator = default_converter(ir_graph, *args, **kwargs, attr_cls=attrs.ConvAttr)
    params = [operator.inputs[1]]
    if len(operator.inputs) > 2:
        params.append(operator.inputs[2])
    mark_var_as_parameter(ir_graph, params)
    return operator


@convert_onnx_operator(OP.CONV_TRANSPOSE)
def convert_conv_transpose(ir_graph, *args, **kwargs):
    operator = default_converter(
        ir_graph, *args, **kwargs, attr_cls=attrs.ConvTransposeAttr
    )
    params = [operator.inputs[1]]
    if len(operator.inputs) > 2:
        params.append(operator.inputs[2])
    mark_var_as_parameter(ir_graph, params)
    return operator


@convert_onnx_operator(OP.COS)
def convert_cos(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.COSH)
def convert_cosh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.CONSTANT)
def convert_constant(*args, **kwargs):
    operator = default_converter(*args, **kwargs, attr_cls=attrs.ConstantAttr)
    if operator.attributes.value is not None:
        value = numpy_helper.to_array(operator.attributes.value)
        if value.ndim == 0:
            value = value.item()

        operator.attributes.value = value
    return operator


@convert_onnx_operator(OP.CONSTANT_OF_SHAPE)
def convert_constant_of_shape(*args, **kwargs):
    def parse_attr(**attr_dict):
        value = attr_dict.get("value", None)
        if value is not None:
            attr_dict["value"] = onnx.numpy_helper.to_array(value)
        return attrs.ConstantOfShape(**attr_dict)

    return default_converter(*args, **kwargs, attr_cls=parse_attr)


@convert_onnx_operator(OP.DEQUANTIZELINEAR)
def convert_dequantizelinear(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.DequantizeLinearAttr)


@convert_onnx_operator(OP.QUANTIZELINEAR)
def convert_quantizelinear(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.QuantizeLinearAttr)


@convert_onnx_operator(OP.DIV)
def convert_div(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.DROPOUT)
def convert_dropout(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.DropoutAttr)


@convert_onnx_operator(OP.ELU)
def convert_elu(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.EluAttr)


@convert_onnx_operator(OP.EQUAL)
def convert_equal(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ERF)
def convert_erf(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.EXP)
def convert_exp(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.EXPAND)
def convert_expand(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.EYELIKE)
def convert_eyelike(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.EyeLikeAttr)


@convert_onnx_operator(OP.FLATTEN)
def convert_flatten(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.FlattenAttr)


@convert_onnx_operator(OP.FLOOR)
def convert_floor(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GATHER)
def convert_gather(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.GATHER_ELES)
def convert_gather_elements(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.GATHER_ND)
def convert_gather_elements(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.GatherNDAttr)


@convert_onnx_operator(OP.GEMM)
def convert_gemm(ir_graph: Graph, *args, **kwargs):
    operator = default_converter(ir_graph, *args, **kwargs, attr_cls=attrs.GemmAttr)
    params = [operator.inputs[1]]
    if len(operator.inputs) > 2:
        params.append(operator.inputs[2])
    mark_var_as_parameter(ir_graph, params)
    return operator


@convert_onnx_operator(OP.GLOBAL_AVG_POOL)
def convert_global_avg_pool(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GLOBAL_MAX_POOL)
def convert_global_max_pool(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GREATER)
def convert_greater(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GREATER_EQUAL)
def convert_greater_equal(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GROUP_NORM)
def convert_group_norm(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GRU)
def convert_gru(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.GruAttr)


@convert_onnx_operator(OP.HARDSIGMOID)
def convert_hardsigmoid(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.HardSigmoidAttr)


@convert_onnx_operator(OP.HARDSWISH)
def convert_hardswish(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.HARDMAX)
def convert_hardmax(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.IDENTITY)
def convert_identity(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.INSTANCE_NORM)
def convert_instance_norm(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.InstanceNormAttr)


@convert_onnx_operator(OP.ISINF)
def convert_isinf(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.ISNAN)
def convert_isnan(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.LAYER_NORM)
def convert_layer_norm(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.LayerNormAttr)


@convert_onnx_operator(OP.LEAKY_RELU)
def convert_leak_relu(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.LeakReluAttr)


@convert_onnx_operator(OP.LESS)
def convert_less(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.LESS_EQUAL)
def convert_less_equal(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.LRN)
def convert_lrn(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.LrnAttr)


@convert_onnx_operator(OP.LSTM)
def convert_lstm(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.LstmAttr)


@convert_onnx_operator(OP.LOG)
def convert_log(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.LOG_SOFTMAX)
def convert_log_softmax(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.MATMUL)
def convert_matmul(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MAX)
def convert_max(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MAX_POOL)
def convert_max_pool(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.MaxPoolAttr)


@convert_onnx_operator(OP.MAX_ROI_POOL)
def convert_max_roi_pool(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MAX_UNPOOL)
def convert_max_unpool(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.MaxUnpoolAttr)


@convert_onnx_operator(OP.MEAN)
def convert_mean(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MIN)
def convert_min(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MISH)
def convert_mish(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MUL)
def convert_mul(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.MOD)
def convert_mod(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.NEG)
def convert_neg(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.NONZERO)
def convert_nonzero(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.NOT)
def convert_not(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.OR)
def convert_or(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.PAD)
def convert_pad(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.PadAttr)


@convert_onnx_operator(OP.POW)
def convert_pow(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.PRELU)
def convert_prelu(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.RANDOM_UNIFORM_LIKE)
def convert_random_uniform_like(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.RandomUniformLikeAttr)


@convert_onnx_operator(OP.RANGE)
def convert_range(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(
    [
        OP.REDUCE_L1,
        OP.REDUCE_L2,
        OP.REDUCE_MAX,
        OP.REDUCE_MIN,
        OP.REDUCE_MEAN,
        OP.REDUCE_PROD,
        OP.REDUCE_SUM,
    ]
)
def convert_reduce_ops(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ReductionAxesAttr)


@convert_onnx_operator(OP.RELU)
def convert_relu(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.RESHAPE)
def convert_reshape(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.RESIZE)
def convert_resize(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ResizeAttr)


@convert_onnx_operator(OP.RNN)
def convert_rnn(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.RnnAttr)


@convert_onnx_operator(OP.ROIALIGN)
def convert_roi_align(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.RoiAlignAttr)


@convert_onnx_operator(OP.ROUND)
def convert_round(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SCATTER)
def convert_scatter(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.SCATTER_ELES)
def convert_scatter_elements(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ScatterElementsAttr)


@convert_onnx_operator(OP.SCATTER_ND)
def convert_scatter_nd(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.GATHER_ND)
def convert_gather_nd(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SELU)
def convert_selu(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.SeluAttr)


@convert_onnx_operator(OP.SHAPE)
def convert_shape(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SHRINK)
def convert_shrink(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.ShrinkAttr)


@convert_onnx_operator(OP.SLICE)
def convert_slice(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SIGMOID)
def convert_sigmoid(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SIGN)
def convert_sign(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SILU)
def convert_silu(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SIN)
def convert_sin(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SINH)
def convert_sinh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SIZE)
def convert_size(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SOFTMAX)
def convert_softmax(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxisAttr)


@convert_onnx_operator(OP.SOFTPLUS)
def convert_softplus(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SOFTSIGN)
def convert_softsign(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SPLIT)
def convert_split(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.SplitAttr)


@convert_onnx_operator(OP.SPLIT_TO_SEQUENCE)
def convert_split_to_sequence(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.SplitToSequenceAttr)


@convert_onnx_operator(OP.SQRT)
def convert_sqrt(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SQUEEZE)
def convert_squeeze(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxesAttr)


@convert_onnx_operator(OP.SUB)
def convert_sub(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.SUM)
def convert_sum(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.TAN)
def convert_tan(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.TANH)
def convert_tanh(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.TILE)
def convert_tile(*args, **kwargs):
    return default_converter(*args, **kwargs)


@convert_onnx_operator(OP.TOPK)
def convert_topk(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.TopkAttr)


@convert_onnx_operator(OP.TRANSPOSE)
def convert_transpose(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.TransposeAttr)


@convert_onnx_operator(OP.UPSAMPLE)
def convert_upsample(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.UpsampleAttr)


@convert_onnx_operator(OP.UNSQUEEZE)
def convert_unsqueeze(*args, **kwargs):
    return default_converter(*args, **kwargs, attr_cls=attrs.AxesAttr)


@convert_onnx_operator(OP.WHERE)
def convert_where(*args, **kwargs):
    return default_converter(*args, **kwargs)
