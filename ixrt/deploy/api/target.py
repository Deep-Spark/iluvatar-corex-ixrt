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

from ixrt.deploy.ir.base_target import TargetList
from ixrt.deploy.quantizer.save_quant_param import SaveQuantParameterPPQStyle

from ..backend.onnx.onnx_target import OnnxTarget
from ..backend.ixrt.onnx_target import IxrtQuantizedOnnxTarget


def create_target(
    saved_path: str,
    example_inputs=None,
    quant_params_path: str = None,
    **export_onnx_kwargs,
):
    targets = [
        IxrtQuantizedOnnxTarget(
            saved_path=saved_path,
            example_inputs=example_inputs,
            **export_onnx_kwargs,
        )
    ]
    if quant_params_path is not None:
        targets.append(
            SaveQuantParameterPPQStyle(
                quant_params_path, json_dump_kwargs={"indent": 4}  # type: ignore
            )
        )

    return TargetList(targets)  # type: ignore
