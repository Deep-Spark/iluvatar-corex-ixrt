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

import os
import uuid
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


ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def adapt_buffer_to_ort_input(buffer, ort_input):
    expected = ORT_TYPE_TO_NUMPY.get(ort_input.type)
    if expected is None or buffer.dtype == expected:
        return buffer
    if buffer.dtype == np.uint16:
        buffer = (buffer.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return buffer.astype(expected)


class OrtLayerSaver:
    def __init__(self, config, input_buffers):
        self.config = config
        self.input_buffers = input_buffers
        self.inference_result = OrderedDict()

    def save(self):
        # Use load_external_data=False so the proto stays small (only the
        # graph structure, not the multi-GB weight blobs). External data
        # files remain on disk and OnnxRuntime resolves them by path.
        raw_onnx = onnx.load(self.config.onnx_path, load_external_data=False)
        # 1. add extend output
        extra = list(self.config.verify_tensors or [])
        if extra:
            existing = {o.name for o in raw_onnx.graph.output}
            for name in extra:
                if name not in existing:
                    raw_onnx.graph.output.extend([onnx.ValueInfoProto(name=name)])
                    existing.add(name)
        elif not self.config.only_verify_outputs:
            for node in raw_onnx.graph.node:
                for output in node.output:
                    raw_onnx.graph.output.extend([onnx.ValueInfoProto(name=output)])
        # 2. Start to infer
        if self.config.ort_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]

        # Save the extended model to a temporary file and let OnnxRuntime
        # load from the file path. This avoids SerializeToString() which
        # fails for models exceeding the protobuf 2 GB limit.
        # Saving to the same directory as the original model ensures
        # external data relative paths resolve correctly.
        model_dir = os.path.dirname(os.path.abspath(self.config.onnx_path))
        tmp_onnx_path = os.path.join(
            model_dir, f".ort_extended_{uuid.uuid4().hex}.onnx"
        )
        try:
            onnx.save(raw_onnx, tmp_onnx_path)
            ort_session = onnxruntime.InferenceSession(
                tmp_onnx_path, providers=providers
            )
        finally:
            if os.path.exists(tmp_onnx_path):
                os.remove(tmp_onnx_path)

        input_buffers = {}
        for input in ort_session.get_inputs():
            input_name = input.name
            input_buffers[input_name] = adapt_buffer_to_ort_input(
                self.input_buffers[input_name], input
            )

        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, input_buffers)
        ort_outs = OrderedDict(zip(outputs, ort_outs))
        for edge_name, v in ort_outs.items():
            if v is not None:
                filename = get_edge_path(self.config.ort, edge_name)
                np.save(filename, v)
                self.inference_result[edge_name] = InferResult(
                    producer_node="ORT_Node", edge_name=edge_name, saved_path=filename
                )
        # also save input for better comparison
        for edge_name, v in input_buffers.items():
            filename = get_edge_path(self.config.ort, edge_name)
            np.save(filename, v)
            self.inference_result[edge_name] = InferResult(
                producer_node="ORT_Node", edge_name=edge_name, saved_path=filename
            )
