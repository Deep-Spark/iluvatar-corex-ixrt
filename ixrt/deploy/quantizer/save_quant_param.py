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

import json
import os
from collections import OrderedDict

from ..ir import BaseTarget
from ..ir.data_type import get_type_bit_width
from .quant_paramter import QuantParameter


class SaveQuantParameterPPQStyle(BaseTarget):
    def __init__(self, saved_path: str = "", json_dump_kwargs: dict = None):
        super(SaveQuantParameterPPQStyle, self).__init__()
        self.saved_path = self._check_saved_path(saved_path)
        self.json_dump_kwargs = dict() if json_dump_kwargs is None else json_dump_kwargs

    def export(self, graph):
        quant_parameters = graph.quant_parameters
        quant_parameters = self._preprocess_quant_params(quant_parameters)
        operator_info = self._collect_operator_info(graph)
        quant_parameters = dict(quant_info=quant_parameters, op_info=operator_info)
        if self.saved_path:
            with open(self.saved_path, "w") as f:
                json.dump(quant_parameters, f, **self.json_dump_kwargs)
        return quant_parameters

    def _preprocess_quant_params(self, quant_params: OrderedDict):
        new_quant_params = OrderedDict()
        for k, param in quant_params.items():
            if isinstance(param, QuantParameter):
                new_quant_params[k] = dict(
                    bit_width=get_type_bit_width(param.qtype),
                    per_channel=param.per_channel,
                    quant_flag=True,
                    sym=param.symmetrical,
                    scale=param.scale,
                    zero_point=param.zero_point,
                    tensor_min=param.tensor_min,
                    tensor_max=param.tensor_max,
                    q_min=param.qtype_min,
                    q_max=param.qtype_max,
                    quant_dim=param.quant_dim,
                )

        return new_quant_params

    def _collect_operator_info(self, graph) -> dict:
        op_info = dict()
        for operator in graph.operators.values():
            if operator.is_quant_operator:
                op_info[operator.name] = dict(data_type="int8")
            else:
                op_info[operator.name] = dict(data_type="float32")
        return op_info

    @classmethod
    def load(cls, graph, path=None, quant_params=None):
        if path is None and quant_params is None:
            raise ValueError("Got invalid quantization file or parameters.")

        op_info = dict()

        if path is not None:
            with open(path) as f:
                qparams_json = json.load(f)

            if "op_info" in qparams_json:
                op_info = qparams_json["op_info"]

            qparams_json = qparams_json["quant_info"]
        else:
            qparams_json = quant_params

            if "op_info" in qparams_json:
                op_info = qparams_json["op_info"]

            if "quant_info" in qparams_json:
                qparams_json = qparams_json["quant_info"]

        for var_name, qparam in qparams_json.items():
            qparam = QuantParameter(
                per_channel=qparam["per_channel"],
                scale=qparam["scale"],
                zero_point=qparam["zero_point"],
                tensor_max=qparam["tensor_max"],
                tensor_min=qparam["tensor_min"],
                qtype_max=qparam["q_max"],
                qtype_min=qparam["q_min"],
                qtype="int8",  # TODO: Support parse more type
                symmetrical=qparam["sym"],
                quant_dim=qparam.get("quant_dim", 0),
            )
            graph.add_quant_parameter(var_name, qparam)

        for op_name, data_type in op_info.items():
            if "data_type" in data_type:
                data_type = data_type["data_type"]

            if not isinstance(data_type, str):
                raise RuntimeError(
                    f"Got invalid data type for operator `{op_name}`, got {data_type}."
                )

            if data_type.lower() in ["uint8", "int8", "int4", "uint4"]:
                graph.get_operator(op_name).mark_as_quant_op()

        return graph

    def _check_saved_path(self, path):
        if path is None:
            return path

        dir = os.path.dirname(path)
        if not os.path.exists(dir) and dir not in ["", ".", "./"]:
            os.makedirs(dir, exist_ok=True)
        return path
