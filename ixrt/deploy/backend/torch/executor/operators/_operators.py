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
import operator as operators
from functools import reduce
from typing import Mapping

import numpy as np
import onnx.numpy_helper
import torch
import torch.nn.functional as F
from ixrt.deploy.backend.onnx.export.utils import (
    convert_onnx_conv_padding_to_torch,
    convert_onnx_pads_to_torch,
)
from ixrt.deploy.ir.data_type_mapping import onnx_to_torch_dtype
from ixrt.deploy.ir.operator_attr import *
from ixrt.deploy.ir.operator_type import OperatorType as OP

from .base import default_call_op, registe_executor_operator


def get_item(x, i, default=None):
    if isinstance(i, str):
        if isinstance(x, Mapping):
            return x.get(i, default)
        if hasattr(x, i):
            return getattr(x, i)
        return default
    if i < len(x):
        return x[i]
    return default


def to_py_type(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 0:
            return x.item()
        return x.tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    return x


def convert_list_to_ints(values):
    new_values = []
    for v in values:
        new_values.append(int(v))
    return type(values)(new_values)


@registe_executor_operator(OP.ABS)
def call_abs(executor, operator, inputs, attr):
    return default_call_op(torch.abs, inputs, attr)


@registe_executor_operator(OP.ACOS)
def call_acos(executor, operator, inputs, attr):
    return default_call_op(torch.acos, inputs, attr)


@registe_executor_operator(OP.ACOSH)
def call_acosh(executor, operator, inputs, attr):
    return default_call_op(torch.acosh, inputs, attr)


@registe_executor_operator(OP.ADD)
def call_add(executor, operator, inputs, attr):
    if inputs[1].ndim == 1:
        inputs[1] = inputs[1].reshape([1] * inputs[1].ndim + [inputs[1].shape[0]])

    return default_call_op(torch.add, inputs, attr)


@registe_executor_operator(OP.AND)
def call_and(executor, operator, inputs, attr):
    return torch.bitwise_and(*inputs)


@registe_executor_operator(OP.ARGMAX)
def call_argmax(executor, operator, inputs, attr: ReductionAxisAttr):
    assert (
        isinstance(attr.keepdims, int) and attr.keepdims <= 1
    ), "Unsupported attribute."
    return torch.argmax(inputs[0], dim=attr.axis, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.ARGMIN)
def call_argmin(executor, operator, inputs, attr):
    assert (
        isinstance(attr.keepdims, int) and attr.keepdims <= 1
    ), "Unsupported attribute."
    return torch.argmin(inputs[0], dim=attr.axis, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.ASIN)
def call_asin(executor, operator, inputs, attr):
    return default_call_op(torch.asin, inputs, attr)


@registe_executor_operator(OP.ASINH)
def call_asinh(executor, operator, inputs, attr):
    return default_call_op(torch.asinh, inputs, attr)


@registe_executor_operator(OP.ATAN)
def call_atan(executor, operator, inputs, attr):
    return default_call_op(torch.atan, inputs, attr)


@registe_executor_operator(OP.ATANH)
def call_atanh(executor, operator, inputs, attr):
    return default_call_op(torch.atanh, inputs, attr)


@registe_executor_operator([OP.AVG_POOL, OP.GLOBAL_AVG_POOL])
def call_avg_pool(executor, operator, inputs, attr: PoolingAttr):
    ndim = inputs[0].ndim
    x = inputs[0]
    if operator.op_type == OP.GLOBAL_AVG_POOL:
        pool_kwargs = dict(input=x, kernel_size=inputs[0].shape[2:], stride=1)
    else:
        x, pads = convert_onnx_conv_padding_to_torch(x, attr.get("pads", [0])[0])
        strides = attr.get("strides", 1)

        pool_kwargs = dict(
            input=x,
            kernel_size=attr.kernel_shape,
            stride=strides,
            padding=pads,
            ceil_mode=bool(attr.ceil_mode),
        )

    if ndim == 3:
        return F.avg_pool1d(**pool_kwargs)
    elif ndim == 4:
        return F.avg_pool2d(**pool_kwargs)
    elif ndim == 5:
        return F.avg_pool3d(**pool_kwargs)
    else:
        raise RuntimeError(f"Not support inputs, got shape {inputs[0].shape}.")


@registe_executor_operator(OP.BATCH_NORM)
def call_batch_norm(executor, operator, inputs, attr: BatchNormAttr):
    return F.batch_norm(
        input=inputs[0],
        weight=get_item(inputs, 1),
        bias=get_item(inputs, 2),
        running_mean=get_item(inputs, 3),
        running_var=get_item(inputs, 4),
        eps=attr.epsilon,
        momentum=attr.momentum,
    )


@registe_executor_operator(OP.CAST)
def call_cast(executor, operator, inputs, attr: CastAttr):
    torch_dtype = onnx_to_torch_dtype(attr.to)
    if torch.is_tensor(inputs[0]):
        x = inputs[0]
    else:
        x = torch.tensor(inputs[0], device=executor.default_device())
    return x.to(dtype=torch_dtype)


@registe_executor_operator(OP.CAST_LIKE)
def call_cast_like(executor, operator, inputs, attr):
    return inputs[0].to(inputs[1])


@registe_executor_operator(OP.CEIL)
def call_ceil(executor, operator, inputs, attr):
    return default_call_op(torch.ceil, inputs, attr)


@registe_executor_operator(OP.CELU)
def call_celuf(executor, operator, inputs, attr: CeluAttr):
    return F.celu(inputs[0], alpha=attr.alpha)


@registe_executor_operator(OP.CLIP)
def call_clip(executor, operator, inputs, attr: ClipAttr):
    if len(inputs) > 1:
        attr.min = to_py_type(inputs[1])
        attr.max = to_py_type(inputs[2])
    return default_call_op(torch.clip, inputs[:1], attr)


@registe_executor_operator(OP.CONCAT)
def call_concat(executor, operator, inputs, attr: AxisAttr):
    for i in range(len(inputs)):
        x = inputs[i]
        if x.ndim == 0:
            inputs[i] = x.unsqueeze(0)
    return torch.cat(inputs, dim=attr.axis)


@registe_executor_operator(OP.CONCAT_FROM_SEQUENCE)
def call_concat_from_sequence(executor, operator, inputs, attr: ConcatFromSequenceAttr):
    assert attr.new_axis in [0, None], "Not support new_axis attribute."
    return torch.stack(inputs, dim=attr.axis)


@registe_executor_operator(OP.CONV)
def call_conv(executor, operator, inputs, attr: ConvAttr):
    x, pads = convert_onnx_conv_padding_to_torch(inputs[0], attr.get("pads", [1]))

    assert attr.get("auto_pad") in ["NOTSET", None], "Not support auto_pad attribute."

    ndim = inputs[0].ndim

    conv_params = dict(
        input=x,
        weight=inputs[1],
        bias=get_item(inputs, 2),
        stride=attr.get("strides", 1),
        padding=pads,
        dilation=attr.get("dilations", 1),
        groups=attr.get("group", 1),
    )

    if ndim == 3:
        return F.conv1d(**conv_params)
    elif ndim == 4:
        return F.conv2d(**conv_params)
    elif ndim == 5:
        return F.conv3d(**conv_params)
    raise RuntimeError(f"Not support inputs, got shape {inputs[0].shape}.")


@registe_executor_operator(OP.CONV_TRANSPOSE)
def call_conv_transpose(executor, operator, inputs, attr: ConvTransposeAttr):
    x, pads = convert_onnx_conv_padding_to_torch(inputs[0], attr.get("pads", [1])[0])

    assert attr.auto_pad in ["NOTSET", None], "Not support auto_pad attribute."

    ndim = inputs[0].ndim

    conv_params = dict(
        input=x,
        weight=inputs[1],
        bias=get_item(inputs, 2),
        stride=attr.get("strides", 1),
        padding=pads,
        dilation=attr.get("dilations", 1),
        groups=attr.get("group", 1),
        output_padding=attr.get("output_padding", 0),
    )

    if ndim == 3:
        return F.conv_transpose1d(**conv_params)
    elif ndim == 4:
        return F.conv_transpose2d(**conv_params)
    elif ndim == 5:
        return F.conv_transpose3d(**conv_params)
    raise RuntimeError(f"Not support inputs, got shape {inputs[0].shape}.")


@registe_executor_operator(OP.COS)
def call_cos(executor, operator, inputs, attr):
    return default_call_op(torch.cos, inputs, attr)


@registe_executor_operator(OP.COSH)
def call_cosh(executor, operator, inputs, attr):
    return default_call_op(torch.cosh, inputs, attr)


@registe_executor_operator(OP.CONSTANT)
def call_constant(executor, operator, inputs, attr: ConstantAttr):
    x = attr.value
    for atype, val in attr.to_dict().items():
        if val is not None:
            x = val
    try:
        out = onnx.numpy_helper.to_array(x)
    except:
        out = x
    return torch.tensor(out).to(device=executor.default_device())


@registe_executor_operator(OP.CONSTANT_OF_SHAPE)
def call_constant_of_shape(executor, operator, inputs, attr: ConstantOfShape):
    v = to_py_type(inputs[0])
    if isinstance(v, (tuple, list)):
        v = v[0]
    return inputs[0].new_full(inputs[0].shape, v)


@registe_executor_operator(OP.DIV)
def call_div(executor, operator, inputs, attr):
    x, y = inputs
    out = x / y

    if torch.is_tensor(x):
        x_dtype = str(x.dtype)
    else:
        x_dtype = type(to_py_type(x)).__name__

    if torch.is_tensor(y):
        y_dtype = str(y.dtype)
    else:
        y_dtype = type(to_py_type(y)).__name__

    if "int" in x_dtype and "int" in y_dtype:
        return out.floor().to(dtype=x.dtype)
    return out


@registe_executor_operator(OP.DEQUANTIZELINEAR)
def call_dequantizelinear(executor, operator, inputs, attr: DequantizeLinearAttr):
    return inputs[0]


@registe_executor_operator(OP.QUANTIZELINEAR)
def call_quantizelinear(executor, operator, inputs, attr: QuantizeLinearAttr):
    return inputs[0]


@registe_executor_operator(OP.DROPOUT)
def call_dropout(executor, operator, inputs, attr: DropoutAttr):
    # TODO: Set training status
    return F.dropout(inputs[0], p=attr.ratio, training=False)


@registe_executor_operator(OP.ELU)
def call_elu(executor, operator, inputs, attr: EluAttr):
    return F.elu(inputs[0], alpha=attr.alpha)


@registe_executor_operator(OP.EQUAL)
def call_equal(executor, operator, inputs, attr):
    return default_call_op(torch.eq, inputs, attr)


@registe_executor_operator(OP.ERF)
def call_erf(executor, operator, inputs, attr):
    return default_call_op(torch.erf, inputs, attr)


@registe_executor_operator(OP.EXP)
def call_exp(executor, operator, inputs, attr):
    return default_call_op(torch.exp, inputs, attr)


@registe_executor_operator(OP.EXPAND)
def call_expand(executor, operator, inputs, attr):
    shape = to_py_type(inputs[1])
    assert isinstance(shape, (tuple, list)), f"Invalid shape {shape}"
    return inputs[0].expand(*shape)


@registe_executor_operator(OP.EYELIKE)
def call_eyelike(executor, operator, inputs, attr: EyeLikeAttr):
    if attr.dtype is not None:
        dtype = onnx_to_torch_dtype(attr.dtype)
    else:
        dtype = inputs[0].dtype
    if attr.k in [0, None] or attr.k < 0:
        s = inputs[0].shape
        out = torch.eye(s[0], s[1])
    else:
        out = torch.eye(attr.k)
    return out.to(device=inputs[0].device, dtype=dtype)


@registe_executor_operator(OP.FLATTEN)
def call_flatten(executor, operator, inputs, attr: FlattenAttr):
    return torch.flatten(inputs[0], attr.axis)


@registe_executor_operator(OP.FLOOR)
def call_floor(executor, operator, inputs, attr):
    return default_call_op(torch.floor, inputs, attr)


@registe_executor_operator(OP.GATHER)
def call_gather(executor, operator, inputs, attr: AxisAttr):
    data, indices = inputs
    indices = torch.tensor(indices, device=data.device)
    if attr.axis in [0, None]:
        if indices.ndim == 0:
            return data[indices]
        return torch.take(data, indices)
    elif indices.ndim == 0:
        index = [slice(None, None, None)] * data.ndim
        index[attr.axis] = indices.cpu().item()
        return data[index]
    else:
        return torch.take_along_dim(data, indices, dim=attr.axis)


@registe_executor_operator(OP.GATHER_ELES)
def call_gather_elements(executor, operator, inputs, attr):
    return torch.gather(inputs[0], dim=attr.axis, index=inputs[1])


@registe_executor_operator(OP.GATHER_ND)
def call_gather_nd(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.GEMM)
def call_gemm(executor, operator, inputs, attr: GemmAttr):
    if attr.transA:
        inputs[0] = inputs[0].transpose(0, 1)
    if attr.transB:
        inputs[1] = inputs[1].transpose(0, 1)

    return attr.alpha * torch.matmul(*inputs[:2]) + attr.beta * get_item(
        inputs, 2, default=0.0
    )


@registe_executor_operator(OP.GREATER)
def call_greater(executor, operator, inputs, attr):
    return inputs[0] > inputs[1]


@registe_executor_operator(OP.GREATER_EQUAL)
def call_greater_equal(executor, operator, inputs, attr):
    raise inputs[0] >= inputs[1]


@registe_executor_operator(OP.GROUP_NORM)
def call_group_norm(executor, operator, inputs, attr: GroupNormAttr):
    return F.group_norm(
        input=inputs[0],
        weight=inputs[1],
        bias=get_item(inputs, 2),
        eps=attr.epsilon,
        num_groups=attr.num_groups,
    )


@registe_executor_operator(OP.GRU)
def call_gru(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.HARDSIGMOID)
def call_hardsigmoid(executor, operator, inputs, attr: HardSigmoidAttr):
    return F.hardsigmoid(inputs[0])


@registe_executor_operator(OP.HARDSWISH)
def call_hardswish(executor, operator, inputs, attr):
    return F.hardswish(inputs[0])


@registe_executor_operator(OP.HARDMAX)
def call_hardmax(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.IDENTITY)
def call_identity(executor, operator, inputs, attr):
    raise inputs[0]


@registe_executor_operator(OP.INSTANCE_NORM)
def call_instance_norm(executor, operator, inputs, attr: InstanceNormAttr):
    return F.instance_norm(
        input=inputs[0],
        weight=get_item(inputs, 1),
        bias=get_item(inputs, 2),
        running_mean=get_item(inputs, 3),
        running_var=get_item(inputs, 4),
        eps=attr.epsilon,
    )


@registe_executor_operator(OP.ISINF)
def call_isinf(executor, operator, inputs, attr):
    return torch.isinf(inputs[0])


@registe_executor_operator(OP.ISNAN)
def call_isnan(executor, operator, inputs, attr):
    return torch.isnan(inputs[0])


@registe_executor_operator(OP.LAYER_NORM)
def call_layer_norm(executor, operator, inputs, attr: LayerNormAttr):
    return F.layer_norm(
        input=inputs[0],
        weight=inputs[1],
        bias=get_item(inputs, 2),
        normalized_shape=inputs[1].shape,
        eps=attr.epsilon,
    )


@registe_executor_operator(OP.LEAKY_RELU)
def call_leaky_relu(executor, operator, inputs, attr: LeakReluAttr):
    return F.leaky_relu(inputs[0], negative_slope=attr.alpha)


@registe_executor_operator(OP.LESS)
def call_less(executor, operator, inputs, attr):
    return inputs[0] < inputs[1]


@registe_executor_operator(OP.LESS_EQUAL)
def call_less_equal(executor, operator, inputs, attr):
    return inputs[0] <= inputs[1]


@registe_executor_operator(OP.LRN)
def call_lrn(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.LSTM)
def call_lstm(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.LOG)
def call_log(executor, operator, inputs, attr):
    return torch.log(inputs[0])


@registe_executor_operator(OP.LOG_SOFTMAX)
def call_log_softmax(executor, operator, inputs, attr: AxisAttr):
    return F.log_softmax(inputs[0], dim=attr.axis)


@registe_executor_operator(OP.MATMUL)
def call_matmul(executor, operator, inputs, attr):
    return torch.matmul(*inputs)


@registe_executor_operator(OP.MAX)
def call_max(executor, operator, inputs, attr):
    assert len(inputs) >= 1
    ret = inputs[0]
    for next in inputs[1:]:
        ret = torch.maximum(ret, next)
    return ret


@registe_executor_operator([OP.MAX_POOL, OP.GLOBAL_MAX_POOL])
def call_max_pool(executor, operator, inputs, attr: MaxPoolAttr):
    ndim = inputs[0].ndim
    return_indices = len(inputs) == 2
    if operator.op_type == OP.GLOBAL_MAX_POOL:
        pool_kwargs = dict(input=inputs[0], kernel_size=inputs[0].shape[2:], stride=1)
    else:
        x, pads = convert_onnx_conv_padding_to_torch(inputs[0], attr.get("pads", [0]))

        pool_kwargs = dict(
            input=x,
            kernel_size=attr.kernel_shape,
            stride=attr.get("strides", attr.kernel_shape),
            padding=pads,
            ceil_mode=bool(attr.ceil_mode),
            dilation=attr.get("dilations", 1),
        )

    if ndim == 3:
        if return_indices:
            return tuple(F.max_pool1d_with_indices(**pool_kwargs))
        return F.max_pool1d(**pool_kwargs)
    elif ndim == 4:
        if return_indices:
            return tuple(F.max_pool2d_with_indices(**pool_kwargs))
        return F.max_pool2d(**pool_kwargs)
    elif ndim == 5:
        if return_indices:
            return tuple(F.max_pool3d_with_indices(**pool_kwargs))
        return F.max_pool3d(**pool_kwargs)
    else:
        raise RuntimeError(f"Not support inputs, got shape {inputs[0].shape}.")


@registe_executor_operator(OP.MAX_ROI_POOL)
def call_max_roi_pool(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.MAX_UNPOOL)
def call_max_unpool(executor, operator, inputs, attr: MaxUnpoolAttr):
    x, pads = convert_onnx_conv_padding_to_torch(inputs[0], attr.get("pads", [1])[0])

    ndim = inputs[0].ndim
    pool_kwargs = dict(
        input=x,
        indices=inputs[1],
        kernel_size=attr.kernel_shape,
        stride=attr.get("strides", attr.kernel_shape),
        padding=pads,
        output_size=get_item(inputs, 2),
    )
    if ndim == 3:
        return F.max_unpool1d(**pool_kwargs)
    elif ndim == 4:
        return F.max_unpool2d(**pool_kwargs)
    elif ndim == 5:
        return F.max_unpool3d(**pool_kwargs)
    raise RuntimeError(f"Not support inputs, got shape {inputs[0].shape}.")


@registe_executor_operator(OP.MEAN)
def call_mean(executor, operator, inputs, attr):
    return sum(*inputs) / len(inputs)


@registe_executor_operator(OP.MIN)
def call_min(executor, operator, inputs, attr):
    assert len(inputs) >= 1
    ret = inputs[0]
    for next in inputs[1:]:
        ret = torch.minimum(ret, next)
    return ret


@registe_executor_operator(OP.MISH)
def call_mish(executor, operator, inputs, attr):
    return F.mish(inputs[0])


@registe_executor_operator(OP.MUL)
def call_mul(executor, operator, inputs, attr):
    return inputs[0] * inputs[1]


@registe_executor_operator(OP.MOD)
def call_mod(executor, operator, inputs, attr):
    return inputs[0] % inputs[1]


@registe_executor_operator(OP.NEG)
def call_neg(executor, operator, inputs, attr):
    return -inputs[0]


@registe_executor_operator(OP.NONZERO)
def call_nonzero(executor, operator, inputs, attr):
    out = torch.nonzero(inputs[0], as_tuple=True)
    return torch.stack(out)


@registe_executor_operator(OP.NOT)
def call_not(executor, operator, inputs, attr):
    return ~inputs[0]


@registe_executor_operator(OP.OR)
def call_or(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.PAD)
def call_pad(executor, operator, inputs, attr: PadAttr):
    x, pads = inputs[:2]
    pads = to_py_type(pads)
    constant_value = to_py_type(get_item(inputs, 2, 0.0))

    pads = convert_onnx_pads_to_torch(pads)

    if attr.mode == "constant":
        return F.pad(x, pads, value=constant_value, mode=attr.mode)
    raise RuntimeError(f"Not support mode `{attr.mode}`.")


@registe_executor_operator(OP.POW)
def call_pow(executor, operator, inputs, attr):
    return torch.pow(*inputs)


@registe_executor_operator(OP.PRELU)
def call_prelu(executor, operator, inputs, attr):
    return F.prelu(*inputs)


@registe_executor_operator(OP.RANDOM_UNIFORM_LIKE)
def call_random_uniform_like(executor, operator, inputs, attr: RandomUniformLikeAttr):
    return inputs[0].uniform_(attr.low, attr.high)[0]


@registe_executor_operator(OP.RANGE)
def call_range(executor, operator, inputs, attr: RangeAttr):
    return torch.range(start=attr.start, end=attr.limit, step=attr.delta)


@registe_executor_operator(OP.REDUCE_L1)
def call_reduce_l1(executor, operator, inputs, attr: ReductionAxesAttr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.REDUCE_L2)
def call_reduce_l2(executor, operator, inputs, attr: ReductionAxesAttr):
    out = torch.norm(inputs[0], dim=attr.axes, keepdim=bool(attr.keepdims))
    if attr.axes is None and bool(attr.keepdims):
        return out.reshape([1] * inputs[0].dim())
    return out


@registe_executor_operator(OP.REDUCE_MAX)
def call_reduce_max(executor, operator, inputs, attr: ReductionAxesAttr):
    if attr.axes is None:
        x = torch.max(inputs[0])
        if attr.keepdims:
            raise RuntimeError("Reduce is not support the behavior.")
        return x

    if len(attr.axes) == 1:
        out = torch.max(inputs[0], dim=attr.axes[0], keepdim=bool(attr.keepdims))
        if isinstance(out, tuple):
            return out.values
        return out
    return inputs[0].amax(dim=attr, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.REDUCE_MIN)
def call_reduce_min(executor, operator, inputs, attr: ReductionAxesAttr):
    if attr.axes is None:
        if attr.keepdims:
            raise RuntimeError("Reduce is not support the behavior.")
        return torch.min(inputs[0])
    if len(attr.axes) == 1:
        out = torch.min(inputs[0], dim=attr.axes[0], keepdim=bool(attr.keepdims))
        if isinstance(out, tuple):
            return out.values
        return out
    return inputs[0].amin(dim=attr, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.REDUCE_MEAN)
def call_reduce_mean(executor, operator, inputs, attr: ReductionAxesAttr):
    if attr.axes is None:
        if attr.keepdims:
            raise RuntimeError("Not support the behavior.")
        return torch.mean(inputs[0])
    return torch.mean(inputs[0], dim=attr.axes, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.REDUCE_PROD)
def call_reduce_prod(executor, operator, inputs, attr: ReductionAxesAttr):
    dim = attr.axes
    if isinstance(attr.axes, (list, tuple)):
        if len(attr.axes) > 1:
            raise RuntimeError("Not support multi-dims.")
        else:
            dim = attr.axes[0]

    if dim is None:
        if attr.keepdims:
            raise RuntimeError("Not support the behavior.")
        return torch.prod(inputs[0])
    return torch.prod(input=inputs[0], dim=dim, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.REDUCE_SUM)
def call_reduce_sum(executor, operator, inputs, attr):
    if attr.axes is None:
        if attr.keepdims:
            raise RuntimeError("Not support the behavior.")
        return torch.sum(inputs[0])
    return torch.sum(inputs[0], dim=attr.axes, keepdim=bool(attr.keepdims))


@registe_executor_operator(OP.RELU)
def call_relu(executor, operator, inputs, attr):
    return torch.relu(inputs[0])


@registe_executor_operator(OP.RESHAPE)
def call_reshape(executor, operator, inputs, attr):
    shape = to_py_type(inputs[1])
    shape = convert_list_to_ints(shape)
    return inputs[0].reshape(shape)


@registe_executor_operator(OP.RESIZE)
def call_resize(executor, operator, inputs, attr: ResizeAttr):
    scales = to_py_type(get_item(inputs, 2))
    sizes = to_py_type(get_item(inputs, 3))

    mode_cvt = dict(cubic="bicubic", linear="bilinear")
    mode = mode_cvt.get(attr.mode, attr.mode)

    align_corners = attr.coordinate_transformation_mode == "align_corners"
    if mode in ["nearest"]:
        align_corners = None

    ndim = inputs[0].shape

    if isinstance(sizes, (tuple, list)):
        if len(sizes) == 0:
            sizes = None
        else:
            if ndim == 2:
                sizes = sizes[1:]
            else:
                if inputs[0].shape[1] == sizes[1]:
                    sizes = sizes[2:]
                else:
                    raise RuntimeError(
                        "Not support resize tensor on the second dimension."
                    )

    if isinstance(scales, (tuple, list)):
        if len(scales) == 0:
            scales = None
        else:
            if scales[0] != 1 and scales[1] != 1:
                raise RuntimeError(f"Not support this scale mode, got {scales}.")

            if ndim == 2:
                scales = scales[1:]
            else:
                scales = scales[2:]

    return F.interpolate(
        input=inputs[0],
        size=sizes,
        scale_factor=scales,
        mode=mode,
        align_corners=align_corners,
    )


@registe_executor_operator(OP.RNN)
def call_rnn(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.ROIALIGN)
def call_roi_align(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.ROUND)
def call_round(executor, operator, inputs, attr):
    return torch.round(inputs[0])


@registe_executor_operator(OP.SCATTER)
def call_scatter(executor, operator, inputs, attr: AxisAttr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.SCATTER_ELES)
def call_scatter_elements(executor, operator, inputs, attr):
    data, indices, updates = inputs
    indices[indices < 0] += data.shape[attr.axis]
    return data.scatter(attr.axis, indices, updates)


@registe_executor_operator(OP.SCATTER_ND)
def call_scatter_nd(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.GATHER_ND)
def call_gather_nd(executor, operator, inputs, attr):
    input_data, indices = inputs

    batch_dims = operator.attributes.get("batch_dims", 0)
    data_rank = len(input_data.shape)
    assert indices.shape[-1] <= data_rank

    num_i = batch_dims
    num_k = len(input_data.shape) - num_i - indices.shape[-1]
    num_idx = indices.shape[-1]

    shape_i = indices.shape[:num_i]
    shape_j = indices.shape[num_i:-1]
    shape_k = input_data.shape[num_i + num_idx :]
    shape_idx = input_data.shape[num_i : num_i + num_idx]

    # indices reshape
    reshaped_indices = indices.reshape(
        *shape_i, -1, num_idx
    )  # shape [i_1, ..., i_b, J, 1]
    # indices tensordot, expand the last dim in indices
    strides = torch.tensor(
        [reduce(operators.mul, shape_idx[i + 1 :], 1) for i in range(num_idx)],
        device=input_data.device,
        dtype=torch.float,
    )
    merged_indices = torch.tensordot(
        reshaped_indices.float(), strides, 1
    )  # shape [i_1, ..., i_b, J]

    # indices expand
    expanded_indices = (
        merged_indices.reshape(*merged_indices.shape, *([1] * num_k))
        .expand(*merged_indices.shape, *shape_k)
        .long()
    )

    # reshape input
    reshaped_input = input_data.reshape(*shape_i, -1, *shape_k)
    output = reshaped_input.gather(batch_dims, expanded_indices)

    # reshaped output
    reshaped_output = output.reshape(*shape_i, *shape_j, *shape_k)
    return reshaped_output


@registe_executor_operator(OP.SELU)
def call_selu(executor, operator, inputs, attr: SeluAttr):
    return F.selu(inputs[0])


@registe_executor_operator(OP.SHAPE)
def call_shape(executor, operator, inputs, attr):
    return torch.tensor(inputs[0].shape, device=inputs[0].device)


@registe_executor_operator(OP.SHRINK)
def call_shrink(executor, operator, inputs, attr):
    # TODO: Impl
    raise NotImplementedError()


@registe_executor_operator(OP.SLICE)
def call_slice(executor, operator, inputs, attr):
    data = inputs[0]
    starts = to_py_type(inputs[1])
    ends = to_py_type(inputs[2])
    axes = to_py_type(get_item(inputs, 3))
    steps = to_py_type(get_item(inputs, 4))

    ndim = inputs[0].ndim
    if ndim == 0:
        return data

    if axes is None:
        axes = range(len(starts))

    axes = list(axes)
    for i in range(len(axes)):
        if axes[i] < 0:
            axes[i] = ndim + i

    if tuple(axes) != tuple(sorted(axes)):
        raise RuntimeError(f"The `axes` must be ordered, got {axes}.")

    if steps is None:
        steps = [1] * len(starts)

    # Ref PPQ
    flip_dims = []
    slices = {}
    for start, end, step, axis in zip(starts, ends, steps, axes):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step
        slices[axis] = slice(int(start), int(end), int(step))

    slices = [slices.get(axis, slice(None, None)) for axis in range(ndim)]

    if len(flip_dims) > 0:
        data = torch.flip(data, flip_dims)

    return data[slices]


@registe_executor_operator(OP.SIGMOID)
def call_sigmoid(executor, operator, inputs, attr):
    return torch.sigmoid(inputs[0])


@registe_executor_operator(OP.SIGN)
def call_sign(executor, operator, inputs, attr):
    return torch.sign(inputs[0])


@registe_executor_operator(OP.SILU)
def call_silu(executor, operator, inputs, attr):
    return F.silu(inputs[0])


@registe_executor_operator(OP.SIN)
def call_sin(executor, operator, inputs, attr):
    return torch.sin(inputs[0])


@registe_executor_operator(OP.SINH)
def call_sinh(executor, operator, inputs, attr):
    return torch.sinh(inputs[0])


@registe_executor_operator(OP.SIZE)
def call_size(executor, operator, inputs, attr):
    return inputs[0].numel()


@registe_executor_operator(OP.SOFTMAX)
def call_softmax(executor, operator, inputs, attr: AxisAttr):
    return torch.softmax(inputs[0], dim=attr.axis)


@registe_executor_operator(OP.SOFTPLUS)
def call_softpuls(executor, operator, inputs, attr):
    return F.softplus(inputs[0])


@registe_executor_operator(OP.SOFTSIGN)
def call_softsign(executor, operator, inputs, attr):
    return F.softsign(inputs[0])


@registe_executor_operator(OP.SPLIT)
def call_split(executor, operator, inputs, attr: SplitAttr):
    dim = attr.axis if attr.axis is not None else 0
    if attr.split is None:
        if len(inputs) <= 1:
            raise RuntimeError("Invalid split, got none.")
        else:
            split = to_py_type(inputs[1])
            attr.split = split
    else:
        split = attr.split
    return torch.split(inputs[0], split, dim=dim)


@registe_executor_operator(OP.SPLIT_TO_SEQUENCE)
def call_split_to_sequence(executor, operator, inputs, attr):
    dim = attr.axis if attr.axis is not None else 0
    if len(inputs) < 2:
        raise RuntimeError("SplitToSequence got invalid `split`.")

    split = to_py_type(inputs[1])

    return torch.split(inputs[0], split, dim=dim)


@registe_executor_operator(OP.SEQUENCE_AT)
def call_sequence_at(executor, operator, inputs, attr):
    if len(inputs) == 1:
        raise RuntimeError("SequenceAt got invalid `index`.")

    idx = to_py_type(inputs[1])

    return inputs[0][idx]


@registe_executor_operator(OP.SQRT)
def call_sqrt(executor, operator, inputs, attr):
    return torch.sqrt(inputs[0])


@registe_executor_operator(OP.SQUEEZE)
def call_squeeze(executor, operator, inputs, attr: AxesAttr):
    if inputs[0].ndim == 0:
        return inputs[0]

    if len(inputs) == 2:
        dims = inputs[1]
    else:
        dims = attr.axes

    if dims is None:
        return inputs[0].squeeze()

    new_shape = []
    for i, si in enumerate(inputs[0].shape):
        if i not in dims:
            new_shape.append(si)

    return torch.reshape(inputs[0], new_shape)


@registe_executor_operator(OP.SUB)
def call_sub(executor, operator, inputs, attr):
    return inputs[0] - inputs[1]


@registe_executor_operator(OP.SUM)
def call_sum(executor, operator, inputs, attr):
    return sum(inputs)


@registe_executor_operator(OP.TAN)
def call_tan(executor, operator, inputs, attr):
    return torch.tan(inputs[0])


@registe_executor_operator(OP.TANH)
def call_tanh(executor, operator, inputs, attr):
    return torch.tanh(inputs[0])


@registe_executor_operator(OP.TILE)
def call_tile(executor, operator, inputs, attr):
    input, repeats = inputs
    repeats = to_py_type(repeats)
    return input.repeat(repeats)


@registe_executor_operator(OP.TOPK)
def call_topk(executor, operator, inputs, attr: TopkAttr):
    x, k = inputs
    k = to_py_type(k)
    if isinstance(k, (tuple, list)):
        if len(k) > 1:
            raise RuntimeError(f"Not support multi-values for attribute `k`, got {k}.")
        k = k[0]

    out = torch.topk(
        input=x,
        k=k,
        dim=attr.axis,
        largest=bool(attr.largest),
        sorted=bool(attr.sorted),
    )

    if len(operator.outputs) == 1:
        return out.values
    return out.values, out.indices


@registe_executor_operator(OP.TRANSPOSE)
def call_transpose(executor, operator, inputs, attr: TransposeAttr):
    return inputs[0].permute(*attr.perm).contiguous()


@registe_executor_operator(OP.UPSAMPLE)
def call_upsample(executor, operator, inputs, attr):
    x, scales = inputs
    scales = to_py_type(scales)
    model_cvt = dict(cubic="bicubic", linear="bilinear")
    return F.interpolate(
        inputs[0], scale_factor=scales, mode=model_cvt.get(attr.mode, attr.mode)
    )


@registe_executor_operator(OP.UNSQUEEZE)
def call_unsqueeze(executor, operator, inputs, attr: AxesAttr):
    ndim = len(inputs[0].shape) + len(attr.axes)
    shape = list(inputs[0].shape)
    new_shape = []
    for d in range(ndim):
        if d in attr.axes:
            new_shape.append(1)
        else:
            new_shape.append(shape.pop(0))
    return torch.reshape(inputs[0], new_shape)


@registe_executor_operator(OP.WHERE)
def call_where(executor, operator, inputs, attr):
    return torch.where(*inputs)
