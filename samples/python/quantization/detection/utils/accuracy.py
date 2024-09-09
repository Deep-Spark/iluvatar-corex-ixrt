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

import json
import os

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ixrt.deploy.api import *
from torch import nn
from tqdm import tqdm

from .coco_common import *

anchors = [
    [1.25000, 1.62500, 2.00000, 3.75000, 4.12500, 2.87500],
    [1.87500, 3.81250, 3.87500, 2.81250, 3.68750, 7.43750],
    [3.62500, 2.81250, 4.87500, 6.18750, 11.65625, 10.18750],
]

device = 0 if torch.cuda.is_available() else "cpu"


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = torch.Tensor([8, 16, 32]).to(device)  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0).to(device) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [
            torch.empty(0).to(device) for _ in range(self.nl)
        ]  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2).to(device)
        )  # shape(nl,na,2)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            # Detect (boxes only)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))

        return (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")  # torch>=0.7 compatibility
        grid = (
            torch.stack((xv, yv), 2).expand(shape) - 0.5
        )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .cuda()
            .expand(shape)
        )
        return grid, anchor_grid


def compute_model_mAP(
    dataloader,
    data_path,
    executor: TorchExecutor,
    quant=False,
    enable=True,
    device=None,
):
    if quant:
        exec_ctx = executor.enable_quant_context
    else:
        exec_ctx = executor.disable_quant_context

    if device is None:
        device = executor.default_device()

    def _compute(graph):
        if not enable:
            return graph

        decoder = Detect(anchors=anchors)
        decoder.eval()

        jdict = []

        for index, (img, label, img_path, shape) in enumerate(
            tqdm(dataloader, desc="VerifyQuantModel")
        ):
            im = torch.from_numpy(img).to(device)
            targets = label
            paths = img_path
            shapes = shape

            batch, _, height, width = im.shape
            targets[:, 2:] *= np.array((width, height, width, height))

            with exec_ctx():
                out = executor.execute_graph(graph, im)
            out = list(out.values())
            out = sorted(out, key=lambda x: -x.shape[-1])

            output = decoder(out)[0]

            pred = non_max_suppression(output, conf_thres=0.001, iou_thres=0.65)

            for idx, det in enumerate(pred):
                img_path = paths[idx]

                predn = det.clone()

                shape = shapes[idx][0]
                scale_boxes(
                    im[idx].shape[1:], predn[:, :4], shape, shapes[idx][1]
                )  # native-space pred

                save_one_json(predn, jdict, img_path, coco80_to_coco91)

        annotation_file = os.path.join(
            data_path, "annotations", "instances_val2017.json"
        )
        anno = COCO(annotation_file)
        pred = anno.loadRes(jdict)

        cocoeval = COCOeval(anno, pred, "bbox")
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

        if quant:
            print("SimulatedQuantmAP:", cocoeval.stats[1])
        else:
            print("FP32 Accuracy:", cocoeval.stats[1])

        return graph

    return _compute
