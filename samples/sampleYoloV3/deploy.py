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

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from os.path import basename, dirname, join

from ixrt.deploy.api import DataType, GraphTransform, create_source, create_target


class YoloV3Transform:
    def __init__(self, graph, custom):
        self.t = GraphTransform(graph)
        self.graph = graph
        self.op_type = "YoloV3Decoder"

    def AddYoloDecoderOp(self, inputs: list, outputs: list, **attributes):
        self.t.make_operator(self.op_type, inputs=inputs, outputs=outputs, **attributes)
        self.graph.get_variable(outputs[0]).dtype = "FLOAT"
        self.graph.outputs[outputs[0]] = self.graph.get_variable(outputs[0])

        # Add a fake Q node
        # q_inputs =outputs + [self.t.make_variable(value=0), self.t.make_variable(value=0)]
        # op=self.t.make_operator("QuantizeLinear", inputs=q_inputs, outputs=self.t.make_variable())
        # self.graph.outputs[op.outputs[0]] = self.graph.get_variable(op.outputs[0])
        return self.graph

    def SetDynamicInput(self):
        self.t.get_variable("images").set_shape([1, 3, "bs", "bs"])
        return self.graph

    def DropUselessOutputs(
        self,
    ):
        original_outputs = ["output", "664", "692"]
        for var_name in original_outputs:
            self.t.delete_output(var_name)
        return self.graph

    def Cleanup(
        self,
    ):
        self.t.cleanup()

        return self.graph


def add_yolov3_decoder(graph, custom, dynamic=False):
    t = YoloV3Transform(graph, custom)
    graph = t.AddYoloDecoderOp(
        inputs=["output_QuantizeLinear_Output"],
        outputs=["decoder_13"],
        anchor=[116, 90, 156, 198, 373, 326],
        num_class=80,
        stride=32,
        faster_impl=1,
    )
    graph = t.AddYoloDecoderOp(
        inputs=["664_QuantizeLinear_Output"],
        outputs=["decoder_26"],
        anchor=[30, 61, 62, 45, 59, 119],
        num_class=80,
        stride=16,
        faster_impl=1,
    )
    graph = t.AddYoloDecoderOp(
        inputs=["692_QuantizeLinear_Output"],
        outputs=["decoder_52"],
        anchor=[10, 13, 16, 30, 33, 23],
        num_class=80,
        stride=8,
        faster_impl=1,
    )
    t.DropUselessOutputs()
    t.Cleanup()
    if dynamic:
        graph = t.SetDynamicInput()
    return graph


def parse_args():
    file = join(
        dirname(__file__),
        "../../data/yolov3/yolov3_without_decoder_qdq.onnx",
    )
    dest = join(
        dirname(__file__),
        "../../data/yolov3/yolov3_with_decoder_qdq.onnx",
    )
    parser = argparse.ArgumentParser("Add yolov3 decoder")
    parser.add_argument("--src", type=str, default=file)
    parser.add_argument("--dest", type=str, default=dest)
    parser.add_argument("--dynamic", action="store_true", default=False)
    parser.add_argument(
        "--custom", action="store_true", default=False, help="Use IxRT plugin"
    )
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_args()
    graph = create_source(config.src)()
    graph = add_yolov3_decoder(graph, config.custom, config.dynamic)
    create_target(saved_path=config.dest).export(graph)
    print("Surged onnx lies on", config.dest)
