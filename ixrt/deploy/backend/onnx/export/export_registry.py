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

from typing import List, Union

from ixrt.deploy.core import Registry

IR_TO_ONNX_REGISTRY = Registry("IrToOnnxRegistry")


def export_onnx_operator(target: Union[str, List[str]], export_fn=None):
    if isinstance(target, str):
        target = [target]

    if export_fn is not None:
        for t in target:
            IR_TO_ONNX_REGISTRY.add_handler(t, export_fn)
        return export_fn

    def _wrap(convert_func):
        for t in target:
            IR_TO_ONNX_REGISTRY.add_handler(t, convert_func)
        return convert_func

    return _wrap


def get_export_registry():
    return IR_TO_ONNX_REGISTRY


def get_exporter(op_type: str, default=None):
    return get_export_registry().get(op_type, default=default)
