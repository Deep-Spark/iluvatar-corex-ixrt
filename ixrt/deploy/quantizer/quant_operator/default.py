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

from ixrt.deploy.ir.operator_type import OperatorType as OP

from .base import (
    quant_activations,
    quant_based_weight_bias_operator,
    quant_double_input_operator,
    quant_single_input_operator,
    quant_pertensor_weight_bias_operator,
    registe_quant_operator,
)


@registe_quant_operator(OP.ABS)
def convert_abs(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ACOS)
def convert_acos(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ACOSH)
def convert_acosh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ADD)
def convert_add(*args, **kwargs):
    return quant_double_input_operator(*args, **kwargs)


@registe_quant_operator(OP.AND)
def convert_and(*args, **kwargs):
    return


@registe_quant_operator(OP.ARGMAX)
def convert_argmax(*args, **kwargs):
    if "quant_outputs" in kwargs:
        kwargs.pop("quant_outputs")
    return quant_single_input_operator(*args, **kwargs, quant_outputs=False)


@registe_quant_operator(OP.ARGMIN)
def convert_argmin(*args, **kwargs):
    if "quant_outputs" in kwargs:
        kwargs.pop("quant_outputs")
    return quant_single_input_operator(*args, **kwargs, quant_outputs=False)


@registe_quant_operator(OP.ASIN)
def convert_asin(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ASINH)
def convert_asinh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ATAN)
def convert_atan(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ATANH)
def convert_atanh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.AVG_POOL)
def convert_avg_pool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.BATCH_NORM)
def convert_batch_norm(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.CAST)
# def convert_cast(*args, **kwargs):
#     pass


# @registe_quant_operator(OP.CAST_LIKE)
# def convert_cast_like(*args, **kwargs):
#     pass


@registe_quant_operator(OP.CEIL)
def convert_ceil(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.CELU)
def convert_celu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.CLIP)
def convert_clip(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator([OP.CONCAT, OP.CONCAT_FROM_SEQUENCE])
def convert_concat(*args, **kwargs):
    return quant_activations(*args, **kwargs, num_activations=None)


@registe_quant_operator([OP.CONV, OP.CONV_TRANSPOSE])
def convert_conv(*args, **kwargs):
    return quant_based_weight_bias_operator(*args, **kwargs)


@registe_quant_operator(OP.COS)
def convert_cos(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.COSH)
def convert_cosh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.CONSTANT)
# def convert_constant(*args, **kwargs):
#     return


# @registe_quant_operator(OP.CONSTANT_OF_SHAPE)
# def convert_constant_of_shape(*args, **kwargs):
#     return


@registe_quant_operator(OP.DIV)
def convert_div(*args, **kwargs):
    return quant_double_input_operator(*args, **kwargs)


@registe_quant_operator(OP.DROPOUT)
def convert_dropout(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ELU)
def convert_elu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.EQUAL)
# def convert_equal(*args, **kwargs):
#     return


@registe_quant_operator(OP.ERF)
def convert_erf(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.EXP)
def convert_exp(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.EXPAND)
def convert_expand(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.EYELIKE)
# def convert_eyelike(*args, **kwargs):
#     return


@registe_quant_operator(OP.FLATTEN)
def convert_flatten(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.FLOOR)
def convert_floor(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator([OP.GATHER, OP.GATHER_ELES, OP.GATHER_ND])
def convert_gather(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.GEMM)
def convert_gemm(*args, **kwargs):
    return quant_pertensor_weight_bias_operator(*args, **kwargs)


@registe_quant_operator(OP.GLOBAL_AVG_POOL)
def convert_global_avg_pool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.GLOBAL_MAX_POOL)
def convert_global_max_pool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.GREATER)
# def convert_greater(*args, **kwargs):
#     return


# @registe_quant_operator(OP.GREATER_EQUAL)
# def convert_greater_equal(*args, **kwargs):
#     return


@registe_quant_operator(OP.GROUP_NORM)
def convert_group_norm(*args, **kwargs):
    return quant_based_weight_bias_operator(*args, **kwargs)


# @registe_quant_operator(OP.GRU)
# def convert_gru(*args, **kwargs):
#     return


@registe_quant_operator(OP.HARDSIGMOID)
def convert_hardsigmoid(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.HARDSWISH)
def convert_hardswish(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.HARDMAX)
def convert_hardmax(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.IDENTITY)
def convert_identity(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.INSTANCE_NORM)
def convert_instance_norm(*args, **kwargs):
    return quant_based_weight_bias_operator(*args, **kwargs)


# @registe_quant_operator(OP.ISINF)
# def convert_isinf(*args, **kwargs):
#     return


# @registe_quant_operator(OP.ISNAN)
# def convert_isnan(*args, **kwargs):
#     return


@registe_quant_operator(OP.LAYER_NORM)
def convert_layer_norm(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.LEAKY_RELU)
def convert_leak_relu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.LESS)
# def convert_less(*args, **kwargs):
#     return


# @registe_quant_operator(OP.LESS_EQUAL)
# def convert_less_equal(*args, **kwargs):
#     return


@registe_quant_operator(OP.LRN)
def convert_lrn(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.LSTM)
# def convert_lstm(*args, **kwargs):
#     return


@registe_quant_operator(OP.LOG)
def convert_log(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.LOG_SOFTMAX)
def convert_log_softmax(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MATMUL)
def convert_matmul(*args, **kwargs):
    return quant_double_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MAX)
def convert_max(*args, **kwargs):
    return quant_activations(*args, **kwargs, num_activations=None)


@registe_quant_operator(OP.MAX_POOL)
def convert_max_pool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MAX_ROI_POOL)
def convert_max_roi_pool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MAX_UNPOOL)
def convert_max_unpool(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MEAN)
def convert_mean(*args, **kwargs):
    return quant_activations(*args, **kwargs, num_activations=None)


@registe_quant_operator(OP.MIN)
def convert_min(*args, **kwargs):
    return quant_activations(*args, **kwargs, num_activations=None)


@registe_quant_operator(OP.MISH)
def convert_mish(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MUL)
def convert_mul(*args, **kwargs):
    return quant_double_input_operator(*args, **kwargs)


@registe_quant_operator(OP.MOD)
def convert_mod(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.NEG)
# def convert_neg(*args, **kwargs):
#     return


@registe_quant_operator(OP.NONZERO)
def convert_nonzero(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.NOT)
# def convert_not(*args, **kwargs):
#     return


# @registe_quant_operator(OP.OR)
# def convert_or(*args, **kwargs):
#     return


# @registe_quant_operator(OP.PAD)
# def convert_pad(*args, **kwargs):
#     return


@registe_quant_operator(OP.POW)
def convert_pow(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.PRELU)
def convert_prelu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.RANDOM_UNIFORM_LIKE)
# def convert_random_uniform_like(*args, **kwargs):
#     return


# @registe_quant_operator(OP.RANGE)
# def convert_range(*args, **kwargs):
#     return


@registe_quant_operator(
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
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.RELU)
def convert_relu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.RESHAPE)
def convert_reshape(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.RESIZE)
def convert_resize(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.RNN)
# def convert_rnn(*args, **kwargs):
#     return


@registe_quant_operator(OP.ROIALIGN)
def convert_roi_align(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.ROUND)
def convert_round(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator([OP.SCATTER, OP.SCATTER_ELES, OP.SCATTER_ND, OP.GATHER_ND])
def convert_scatter(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SELU)
def convert_selu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.SHAPE)
# def convert_shape(*args, **kwargs):
#     return


# @registe_quant_operator(OP.SHRINK)
# def convert_shrink(*args, **kwargs):
#     return


@registe_quant_operator(OP.SLICE)
def convert_slice(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SIGMOID)
def convert_sigmoid(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SIGN)
def convert_sign(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SILU)
def convert_silu(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SIN)
def convert_sin(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SINH)
def convert_sinh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.SIZE)
# def convert_size(*args, **kwargs):
#     return


@registe_quant_operator(OP.SOFTMAX)
def convert_softmax(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SOFTPLUS)
def convert_softplus(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SOFTSIGN)
def convert_softsign(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SPLIT)
def convert_split(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SPLIT_TO_SEQUENCE)
def convert_split_to_sequence(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SQRT)
def convert_sqrt(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SQUEEZE)
def convert_squeeze(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SUB)
def convert_sub(*args, **kwargs):
    return quant_double_input_operator(*args, **kwargs)


@registe_quant_operator(OP.SUM)
def convert_sum(*args, **kwargs):
    return quant_activations(*args, **kwargs, num_activations=None)


@registe_quant_operator(OP.TAN)
def convert_tan(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.TANH)
def convert_tanh(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.TILE)
def convert_tile(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.TOPK)
def convert_topk(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.TRANSPOSE)
def convert_transpose(*args, **kwargs):
    return quant_based_weight_bias_operator(*args, **kwargs)


@registe_quant_operator(OP.UPSAMPLE)
def convert_upsample(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


@registe_quant_operator(OP.UNSQUEEZE)
def convert_unsqueeze(*args, **kwargs):
    return quant_single_input_operator(*args, **kwargs)


# @registe_quant_operator(OP.WHERE)
# def convert_where(*args, **kwargs):
#     return
