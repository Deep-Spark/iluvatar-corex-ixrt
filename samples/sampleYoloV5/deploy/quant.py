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

import argparse
import os
import random
from random import shuffle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ixrt.deploy import static_quantize


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str, default="yolov5s_with_decoder.onnx")
    parser.add_argument("--dataset_dir", type=str, default="./coco2017/val2017")
    parser.add_argument(
        "--observer",
        type=str,
        choices=["hist_percentile", "percentile", "minmax", "entropy", "ema"],
        default="hist_percentile",
    )
    parser.add_argument("--disable_quant_names", nargs="+", type=str)
    parser.add_argument("--save_dir", type=str, help="save path", default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quant_format", type=str, default="ppq")
    args = parser.parse_args()
    return args


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        im = cv2.resize(im, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im1 = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return im1, r, (dw, dh)


def getdataloader(dir, step=20, batch_size=32):
    num = step * batch_size
    val_list = [os.path.join(dir, x) for x in os.listdir(dir)]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    dataloader = []
    imgsz = (640, 640)
    for file_path in pic_list:
        pic_data = cv2.imread(file_path)
        org_img = pic_data
        assert org_img is not None, "Image not Found " + file_name
        h0, w0 = org_img.shape[:2]
        inputsz = imgsz[0]
        r = inputsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(
                org_img, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        else:
            image = org_img.copy()

        img, ratio, dwdh = letterbox(
            image, new_shape=[640, 640], auto=False, scaleup=False
        )
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array
        img = torch.from_numpy(img).float()

        dataloader.append(img)

    calibration_dataset = dataloader
    calibration_dataloader = DataLoader(
        calibration_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    return calibration_dataloader


args = parse_args()
setseed(args.seed)
model_name = args.model_name

out_dir = args.save_dir
dataloader = getdataloader(args.dataset_dir, args.step, args.bsz)
print("disable_quant_names : ", args.disable_quant_names)
static_quantize(
    args.model,
    calibration_dataloader=dataloader,
    save_quant_onnx_path=os.path.join(out_dir, f"quantized_{model_name}.onnx"),
    save_quant_params_path=os.path.join(out_dir, f"quantized_{model_name}.json"),
    observer=args.observer,
    data_preprocess=lambda x: x.to("cuda"),
    disable_quant_names=args.disable_quant_names,
    quant_format=args.quant_format,
)
