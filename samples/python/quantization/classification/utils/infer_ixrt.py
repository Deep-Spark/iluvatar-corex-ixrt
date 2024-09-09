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

import os.path

import numpy as np
import torch
from ixrt.deploy import create_target
from tqdm import tqdm


def infer_by_ixrt(
    dataloader, model_path=None, quant_params_path=None, enable=True, work_dir="./"
):
    engine_path = os.path.join(work_dir, "ixrt-infer-tmp.engine")

    def _compute(graph=None):
        if not enable:
            return graph

        if isinstance(graph, (tuple, list)):
            graph = graph[0]

        if graph is not None:
            create_target(model_path, quant_params_path=quant_params_path).export(graph)

        example_inputs = next(iter(dataloader))[0]

        from ixrt import IxRT

        runtime = IxRT.from_onnx(model_path, quant_params_path)

        config = runtime.GetConfig()
        INPUT_SHAPE = [
            example_inputs.shape[0],
            example_inputs.shape[1],
            example_inputs.shape[2],
            example_inputs.shape[3],
        ]

        config.input_shapes = [(config.input_shapes[0][0], INPUT_SHAPE)]
        config.device_idx = 0
        runtime.SetConfig(config)

        runtime.BuildEngine()
        runtime.SerializeEngine(engine_path)

        runtime = IxRT()
        runtime.LoadEngine(engine_path, example_inputs.shape[0])
        print(runtime.GetConfig())

        correct = 0
        for x, y in tqdm(dataloader, desc="VerifyIxRTModel"):
            x = x.numpy()
            y = y.numpy()

            input_map = runtime.GetInputShape()
            output_map = runtime.GetOutputShape()
            input_io_buffers = []
            output_io_buffers = []
            for name, shape in input_map.items():
                _shape, _padding = shape.dims, shape.padding
                _shape = [i + j for i, j in zip(_shape, _padding)]
                _shape = [_shape[0], *_shape[2:], _shape[1]]
                input_io_buffers.append([name, x, shape])

            for name, shape in output_map.items():
                buffer = np.zeros(shape.dims, dtype=np.float32)
                output_io_buffers.append([name, buffer, shape])

            runtime.LoadInput(input_io_buffers)
            runtime.Execute()
            runtime.FetchOutput(output_io_buffers)

            out = output_io_buffers[0][1]
            out = out.argmax(axis=1)
            out = out[: y.shape[0]]
            correct += (y == out).sum()

        acc1 = correct / len(dataloader.dataset)
        print("IxRT Accuracy:", acc1)
        return acc1

    return _compute


def verify_quantized_model(args, dataloader):
    infer_by_ixrt(
        model_path=args.model,
        quant_params_path=args.quant_params,
        dataloader=dataloader,
        enable=True,
    )()
