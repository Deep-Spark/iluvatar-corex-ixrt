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
import sys

from ixrt.deploy.api import DataType, GraphTransform, create_source, create_target


class YoloXTransform:
    def __init__(self, graph):
        self.t = GraphTransform(graph)
        self.graph = graph

    def AddYoloDecoderOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "YoloxDecoder_IXRT", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        # The end of original onnx is output, delete it
        # for var_name in inputs:
        #     self.t.delete_output(var_name)

        return self.graph

    def AddConcatOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "Concat", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddNMSOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "NMS_IXRT", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        for var_name in outputs:
            self.t.add_output(var_name)
            self.t.get_variable(var_name).dtype = "FLOAT"
        return self.graph

    def DropUselessOutputs(
        self,
    ):
        original_outputs = ["output"]
        for var_name in original_outputs:
            self.t.delete_output(var_name)
        return self.graph

    def Cleanup(
        self,
    ):
        self.t.cleanup()

        return self.graph


def add_yolox_postprocess_nodes(graph, config):
    t = YoloXTransform(graph)
    graph = t.AddYoloDecoderOp(
        name="decoder_400",
        inputs=["1096", "1093", "1095"],
        outputs=["decoder_400_boxes", "decoder_400_scores"],
        num_class=80,
        stride=32,
    )
    graph = t.AddYoloDecoderOp(
        name="decoder_1600",
        inputs=["1070", "1067", "1069"],
        outputs=["decoder_1600_boxes", "decoder_1600_scores"],
        num_class=80,
        stride=16,
    )
    graph = t.AddYoloDecoderOp(
        name="decoder_6400",
        inputs=["1044", "1041", "1043"],
        outputs=["decoder_6400_boxes", "decoder_6400_scores"],
        num_class=80,
        stride=8,
    )

    graph = t.AddConcatOp(
        name="concat_boxes",
        inputs=["decoder_6400_boxes", "decoder_1600_boxes", "decoder_400_boxes"],
        outputs=["boxes"],
        axis=3,
    )

    graph = t.AddConcatOp(
        name="concat_scores",
        inputs=["decoder_6400_scores", "decoder_1600_scores", "decoder_400_scores"],
        outputs=["scores"],
        axis=2,
    )

    graph = t.AddNMSOp(
        name="nms",
        inputs=["boxes", "scores"],
        outputs=[
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ],
        share_location=1,
        iou_threshold=0.45,
        score_threshold=0.7,
        max_output_boxes=1000,
        background_class=-1,
    )

    graph = t.DropUselessOutputs()
    graph = t.Cleanup()

    return graph


def add_yolox_qdq_postprocess_nodes(graph, config):
    t = YoloXTransform(graph)
    graph = t.AddYoloDecoderOp(
        name="decoder_400",
        inputs=[
            "1096_DequantizeLinear_Output",
            "1093_DequantizeLinear_Output",
            "1095_DequantizeLinear_Output",
        ],
        outputs=["decoder_400_boxes", "decoder_400_scores"],
        num_class=80,
        stride=32,
    )
    graph = t.AddYoloDecoderOp(
        name="decoder_1600",
        inputs=[
            "1070_DequantizeLinear_Output",
            "1067_DequantizeLinear_Output",
            "1069_DequantizeLinear_Output",
        ],
        outputs=["decoder_1600_boxes", "decoder_1600_scores"],
        num_class=80,
        stride=16,
    )
    graph = t.AddYoloDecoderOp(
        name="decoder_6400",
        inputs=[
            "1044_DequantizeLinear_Output",
            "1041_DequantizeLinear_Output",
            "1043_DequantizeLinear_Output",
        ],
        outputs=["decoder_6400_boxes", "decoder_6400_scores"],
        num_class=80,
        stride=8,
    )

    graph = t.AddConcatOp(
        name="concat_boxes",
        inputs=["decoder_6400_boxes", "decoder_1600_boxes", "decoder_400_boxes"],
        outputs=["boxes"],
        axis=3,
    )

    graph = t.AddConcatOp(
        name="concat_scores",
        inputs=["decoder_6400_scores", "decoder_1600_scores", "decoder_400_scores"],
        outputs=["scores"],
        axis=2,
    )

    graph = t.AddNMSOp(
        name="nms",
        inputs=["boxes", "scores"],
        outputs=[
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ],
        share_location=1,
        iou_threshold=0.45,
        score_threshold=0.7,
        max_output_boxes=1000,
        background_class=-1,
    )

    graph = t.DropUselessOutputs()
    graph = t.Cleanup()

    return graph


def parse_args():
    src_path = "../../data/yolox_m/yolox_m_qdq_quant.onnx"
    dest_path = "../../data/yolox_m/yolox_m_with_decoder_nms.onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--src", help="The exported to ONNX YOLOX-m", type=str, default=src_path
    )
    parser.add_argument(
        "-o",
        "--dest",
        help="The output ONNX model file to write",
        type=str,
        default=dest_path,
    )
    parser.add_argument(
        "-m",
        "--max_output_boxes",
        help="Max output boxes for the NMS operation",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-st",
        "--score_threshold",
        help="The scalar threshold for score",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "-it",
        "--iou_threshold",
        help="The scalar threshold for IOU",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "-q",
        "--with_qdq",
        help="The onnx model including Q/DQ nodes.",
        action="store_true",
    )

    args = parser.parse_args()
    if not all(
        [
            args.src,
            args.dest,
            args.max_output_boxes,
            args.score_threshold,
            args.iou_threshold,
        ]
    ):
        parser.print_help()
        print(
            "\nThese arguments are required: --src --dest --max_output_boxes --score_threshold --iou_threshold"
        )
        sys.exit(1)
    return args


if __name__ == "__main__":
    config = parse_args()
    graph = create_source(config.src)()
    if config.with_qdq:
        graph = add_yolox_qdq_postprocess_nodes(graph, config)
    else:
        graph = add_yolox_postprocess_nodes(graph, config)
    create_target(saved_path=config.dest).export(graph)
    print("Added post-processing layers onnx lies on", config.dest)
