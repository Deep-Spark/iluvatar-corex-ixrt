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

import inspect
import onnx
import torch
from torch import nn


def rename_model_edges(model_path):
   
    model = onnx.load(model_path)

    for node in model.graph.node:
     
        for i, input_name in enumerate(node.input):
            if '.' in input_name:
                node.input[i] = input_name.replace('.', '_')

        for i, output_name in enumerate(node.output):
            if '.' in output_name:
                node.output[i] = output_name.replace('.', '_')

    for initializer in model.graph.initializer:
        if '.' in initializer.name:
            initializer.name = initializer.name.replace('.', '_')

    new_model_path = model_path.replace('.onnx', '_renamed.onnx')
    onnx.save(model, new_model_path)
    return new_model_path


def export(save_path, model: nn.Module, example_inputs, **kwargs):
    if torch.is_tensor(example_inputs):
        example_inputs = (example_inputs,)

    script_kwargs = {}
    if "script_kwargs" in kwargs:
        script_kwargs = kwargs.pop("script_kwargs")

    export_kwargs = {}
    if "export_kwargs" in kwargs:
        export_kwargs = kwargs.pop("export_kwargs")

    model.eval()
    if isinstance(example_inputs, tuple):
        y = model(*example_inputs)
    else:
        y = model(example_inputs)

    script_fn_arg_names = inspect.getfullargspec(torch.jit.script).args
    export_fn_arg_names = inspect.getfullargspec(torch.onnx.export).args

    if "example_inputs" in script_fn_arg_names:
        script_kwargs["example_inputs"] = example_inputs

    if "example_outputs" in script_fn_arg_names:
        script_kwargs["example_outputs"] = y

    if "example_outputs" in export_fn_arg_names:
        export_kwargs["example_outputs"] = y

    if "opset_version" not in export_kwargs:
        export_kwargs["opset_version"] = 11

    if not isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        model = torch.jit.script(model, **script_kwargs)
    torch.onnx.export(model, example_inputs, save_path, **export_kwargs, **kwargs)
