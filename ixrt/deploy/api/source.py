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

import warnings
from typing import Union
import torch.nn

from ..backend.onnx.onnx_source import OnnxSource
from ..backend.torch.onnx import export, rename_model_edges



def create_source(
    model: Union[str, torch.nn.Module],
    example_inputs=None,
    cvt_onnx_kwargs: dict = None,
    rename_edges: bool = False, 
    **kwargs,
):
    if isinstance(model, (torch.nn.Module, torch.jit.ScriptFunction)):
        tmp_saved_path = f"{model.__class__.__name__}.onnx"

        if "cvt_torch_kwargs" in kwargs:
            warnings.warn("`cvt_torch_kwargs` is deprecated and has no effect.")
            kwargs.pop("cvt_torch_kwargs")

        if "use_jit_export" in kwargs:
            warnings.warn("`use_jit_export` is deprecated and has no effect.")
            kwargs.pop("use_jit_export")

        export(
            tmp_saved_path,
            model,
            example_inputs=example_inputs,
            **kwargs,
        )

        model = tmp_saved_path
    if rename_edges: 
        model = rename_model_edges(model)
    source = OnnxSource(model, load_kwargs=cvt_onnx_kwargs)
    return source
