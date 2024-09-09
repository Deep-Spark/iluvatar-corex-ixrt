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

import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True
    return satisfied


def preprocess_img(img_path, img_sz):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, Image.BILINEAR)
    img = F.center_crop(img, img_sz)
    img = F.to_tensor(img)
    img = F.normalize(
        img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
    )
    # img = img.permute(1, 2, 0) NCHW->NHWC
    # NCHW Format
    img = np.asarray(img, dtype="float32")
    return img


def get_dataloader(datasets_dir, bsz, imgsz, label_file_name="val_map.txt"):
    label_file = os.path.join(datasets_dir, label_file_name)
    with open(label_file, "r") as f:
        label_data = f.readlines()
    label_dict = dict()
    for line in label_data:
        line = line.strip().split("\t")
        label_dict[line[0]] = int(line[1])

    files = os.listdir(datasets_dir)
    batch_img, batch_label = [], []

    for file in files:
        if file == label_file_name:
            continue
        file_path = os.path.join(datasets_dir, file)
        img = preprocess_img(file_path, imgsz)
        batch_img.append(np.expand_dims(img, 0))
        batch_label.append(label_dict[file])
        if len(batch_img) == bsz:
            yield np.concatenate(batch_img, 0), np.array(batch_label)
            batch_img, batch_label = [], []

    if len(batch_img) > 0:
        yield np.concatenate(batch_img, 0), np.array(batch_label)


def eval_batch(batch_score, batch_label):
    batch_score = torch.from_numpy(batch_score)
    values, indices = batch_score.topk(5)
    top1, top5 = 0, 0
    for idx, label in enumerate(batch_label):

        if label == indices[idx][0]:
            top1 += 1
        if label in indices[idx]:
            top5 += 1
    return top1, top5
