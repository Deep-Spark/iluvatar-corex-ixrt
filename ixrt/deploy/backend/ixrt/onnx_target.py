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
from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from ixrt.deploy.backend.onnx.onnx_target import OnnxTarget
from ixrt.deploy.ir import Graph
from ixrt.deploy.quantizer.save_quant_param import SaveQuantParameterPPQStyle

from ..onnx.quant_parameter_serializer import pack_quant_params

__all__ = ["IxrtQuantizedOnnxTarget"]

def modify_opset(model, opset_version):
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":  # 默认的 ONNX opset
            opset.version = opset_version

    return model
class IxrtQuantizedOnnxTarget(OnnxTarget):
    def export(self, graph: Graph):
        origin_saved_path = self.saved_path
        self.saved_path = None

        onnx_model = super().export(graph)
        onnx_model = modify_opset(onnx_model, 13)


        if not hasattr(onnx_model.graph, "quantization_annotation"):
            raise RuntimeError(
                "Onnx is not support write quantization_annotation "
                "to protobuf, please upgrade onnx."
            )

        self._add_quantization_params(onnx_model, graph)

        if origin_saved_path:
            onnx.save(onnx_model, origin_saved_path, **self.save_onnx_kwargs)

        return onnx_model

    def _add_quantization_params(self, onnx_model: onnx.ModelProto, graph: Graph):
        quantized_params = SaveQuantParameterPPQStyle(saved_path=None).export(graph)
        pack_quant_params(
            onnx_model.graph.quantization_annotation, quantized_params["quant_info"]
        )
