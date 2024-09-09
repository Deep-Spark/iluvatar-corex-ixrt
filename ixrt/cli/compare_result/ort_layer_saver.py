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

from collections import OrderedDict

import numpy as np

from .infer_result import InferResult
from .utils import get_edge_path

try:
    import onnx
    import onnxruntime

except Exception as e:
    raise ModuleNotFoundError(
        "Please install onnx/onnxruntime/scipy first! pip3 install onnx onnxruntime scipy"
    )

__all__ = ["OrtLayerSaver"]


class OrtLayerSaver:
    def __init__(self, config, input_buffers):
        self.config = config
        self.input_buffers = input_buffers
        self.inference_result = OrderedDict()

    def save(self):
        raw_onnx = onnx.load(self.config.onnx_path)
        # 1. add extend output
        for node in raw_onnx.graph.node:
            for output in node.output:
                raw_onnx.graph.output.extend([onnx.ValueInfoProto(name=output)])
        # 2. Start to infer
        if self.config.ort_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(
            raw_onnx.SerializeToString(), providers=providers
        )

        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, self.input_buffers)
        ort_outs = OrderedDict(zip(outputs, ort_outs))
        for edge_name, v in ort_outs.items():
            if v is not None:
                filename = get_edge_path(self.config.ort, edge_name)
                np.save(filename, v)
                self.inference_result[edge_name] = InferResult(
                    producer_node="ORT_Node", edge_name=edge_name, saved_path=filename
                )
        # also save input for better comparison
        for edge_name, v in self.input_buffers.items():
            filename = get_edge_path(self.config.ort, edge_name)
            np.save(filename, v)
            self.inference_result[edge_name] = InferResult(
                producer_node="ORT_Node", edge_name=edge_name, saved_path=filename
            )
