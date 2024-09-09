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

from collections import defaultdict

import onnx
from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from ixrt.deploy.core import BaseSerializer, Registry

__all__ = ["QuantParameterSerializer", "pack_quant_params", "unpack_quant_params"]

_PACK_DATAFIELD_REGISTRY = Registry("PackDataFieldRegistry")
_UNPACK_DATAFIELD_REGISTRY = Registry("UnpackDataFieldRegistry")


def pack_vector(value: list):
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("pack_vector only support list or tuple.")

    return ",".join([str(v) for v in value])


@_PACK_DATAFIELD_REGISTRY.registe(alias=bool)
def pack_bool_type(value):
    return str(int(value))


@_UNPACK_DATAFIELD_REGISTRY.registe(alias=bool)
def unpack_bool_type(value: str, map_dtype=None):
    return bool(int(value))


@_PACK_DATAFIELD_REGISTRY.registe(alias=float)
def pack_float_type(value):
    MAX_FLOAT_VALUE = 9999999999

    if value == float("inf"):
        value = MAX_FLOAT_VALUE

    if value == -float("inf"):
        value = -MAX_FLOAT_VALUE

    return str(value)


@_UNPACK_DATAFIELD_REGISTRY.registe(alias=float)
def unpack_float_type(value: str, map_dtype=None):
    return float(value)


@_UNPACK_DATAFIELD_REGISTRY.registe(alias=int)
def unpack_int_type(value: str, map_dtype=None):
    return int(value)


@_PACK_DATAFIELD_REGISTRY.registe(alias=list)
def pack_list_type(value):
    return pack_vector(value)


@_UNPACK_DATAFIELD_REGISTRY.registe(alias=list)
def unpack_vector(value: str, map_dtype=None):
    if not isinstance(value, str):
        return value

    if map_dtype is None:
        map_dtype = lambda x: x

    return [map_dtype(x) for x in value.split(",")]


@_PACK_DATAFIELD_REGISTRY.registe(alias=tuple)
def pack_tuple_type(value):
    return pack_vector(value)


@_PACK_DATAFIELD_REGISTRY.registe(alias=dict)
def pack_dict_type(value):
    raise ValueError(f"Not support pack dict type, got {value}.")


@_UNPACK_DATAFIELD_REGISTRY.registe(alias=dict)
def unpack_vector(value: str, map_dtype=None):
    raise ValueError(f"Not support unpack dict type, got {value}.")


def pack_value(value):
    if _PACK_DATAFIELD_REGISTRY.containe(type(value)):
        return _PACK_DATAFIELD_REGISTRY.get(type(value))(value)
    return str(value)


def unpack_value(value, dtype, map_dtype=None):
    if _UNPACK_DATAFIELD_REGISTRY.containe(dtype):
        return _UNPACK_DATAFIELD_REGISTRY.get(dtype)(value, map_dtype)
    raise RuntimeError(f"Cannot unpack {value}.")


class QuantParameterSerializer(BaseSerializer):
    def __init__(self, quantization_annotation: RepeatedCompositeFieldContainer = None):
        self.quantization_annotation = quantization_annotation

    def pack(self, quant_params: dict):
        """
        quantized_params: Dict[str, Dict[str, Any]], Dict[tensor_name, Dict[param_key, param_value]]
        """
        if self.quantization_annotation is None:
            raise ValueError("quantization_annotation must be given, got none.")

        for tensor_name, p in quant_params.items():
            anno = onnx.TensorAnnotation()
            anno.tensor_name = tensor_name

            for field, value in p.items():
                kv_filed = onnx.StringStringEntryProto()
                kv_filed.key = field
                kv_filed.value = pack_value(value)
                anno.quant_parameter_tensor_names.append(kv_filed)

            self.quantization_annotation.append(anno)

        return self.quantization_annotation

    def unpack(self, quantization_annotation: RepeatedCompositeFieldContainer):
        quant_params = defaultdict(dict)

        def switch_case(tensor_name, name, field, dtype, map_dtype=None):
            if field.key == name:
                quant_params[tensor_name][name] = unpack_value(
                    field.value, dtype, map_dtype
                )

        for info in quantization_annotation:
            for field in info.quant_parameter_tensor_names:
                switch_case(info.tensor_name, "bit_width", field, int)
                switch_case(info.tensor_name, "per_channel", field, bool)
                switch_case(info.tensor_name, "quant_flag", field, bool)
                switch_case(info.tensor_name, "sym", field, bool)
                switch_case(info.tensor_name, "scale", field, list, map_dtype=float)
                switch_case(
                    info.tensor_name, "zero_point", field, list, map_dtype=float
                )
                switch_case(
                    info.tensor_name, "tensor_min", field, list, map_dtype=float
                )
                switch_case(
                    info.tensor_name, "tensor_max", field, list, map_dtype=float
                )
                switch_case(info.tensor_name, "q_min", field, int)
                switch_case(info.tensor_name, "q_max", field, int)
                switch_case(info.tensor_name, "quant_dim", field, int)

            tensor_quant_param = quant_params[info.tensor_name]
            if not tensor_quant_param["per_channel"]:
                convert_scalar_keys = [
                    "scale",
                    "zero_point",
                    "tensor_min",
                    "tensor_max",
                ]
                for k in convert_scalar_keys:
                    if (
                        isinstance(tensor_quant_param[k], (tuple, list))
                        and len(tensor_quant_param[k]) == 1
                    ):
                        quant_params[info.tensor_name][k] = tensor_quant_param[k][0]

        return quant_params


def pack_quant_params(
    quantization_annotation: RepeatedCompositeFieldContainer, quant_params: dict
):
    """
    该函数用于将量化参数 (quant_params) 序列化进入到 quantization_annotation 中
    :param quantization_annotation: onnx.GraphProto.quantization_annotation
    :param quant_params: 量化参数
        quantized_params: Dict[str, Dict[str, Any]],
        Dict[tensor_name, Dict[param_key, param_value]]
    :return: RepeatedCompositeFieldContainer
    """
    return QuantParameterSerializer(quantization_annotation).pack(quant_params)


def unpack_quant_params(
    quantization_annotation: RepeatedCompositeFieldContainer,
) -> dict:
    """
    解析量化参数
    :param quantization_annotation: onnx.GraphProto.quantization_annotation
    :return: 量化参数
        Dict[str, Dict[str, Any]],
        Dict[tensor_name, Dict[param_key, param_value]]
    """
    return QuantParameterSerializer().unpack(quantization_annotation)
