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

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from .coco_common import *


class CalibrationCOCO(Dataset):
    def __init__(
        self,
        image_dir_path,
        label_json_path,
        image_size=640,
        stride=32,
        val_mode=True,
        pad_color=114,
    ):

        self.image_dir_path = image_dir_path
        self.label_json_path = label_json_path
        self.image_size = image_size
        self.stride = stride
        self.val_mode = val_mode
        self.pad_color = pad_color

        self.coco = COCO(annotation_file=self.label_json_path)
        if self.val_mode:
            self.img_ids = list(sorted(self.coco.imgs.keys()))  # 5000
        else:  # train mode need images with labels
            self.img_ids = sorted(list(self.coco.imgToAnns.keys()))  # 4952

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # load image
        img_path = self._get_image_path(index)
        img, (h0, w0), (h, w) = self._load_image(index)

        # letterbox
        img, ratio, pad = letterbox(
            img, self.image_size, color=(self.pad_color, self.pad_color, self.pad_color)
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load label
        raw_label = self._load_json_label(index)
        # normalized xywh to pixel xyxy format
        raw_label[:, 1:] = xywhn2xyxy(
            raw_label[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
        )

        raw_label[:, 1:] = xyxy2xywhn(
            raw_label[:, 1:], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
        )

        nl = len(raw_label)  # number of labels
        labels_out = np.zeros((nl, 6))
        labels_out[:, 1:] = raw_label

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array

        return img, labels_out, img_path, shapes

    def _get_image_path(self, index):
        idx = self.img_ids[index]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.image_dir_path, path)
        return img_path

    def _load_image(self, index):
        img_path = self._get_image_path(index)

        im = cv2.imread(img_path)  # BGR
        h0, w0 = im.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(
                im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR
            )
        return (
            im.astype("float32"),
            (h0, w0),
            im.shape[:2],
        )  # im, hw_original, hw_resized

    def _load_json_label(self, index):
        _, (h0, w0), _ = self._load_image(index)

        idx = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        targets = self.coco.loadAnns(ids=ann_ids)

        labels = []
        for target in targets:
            cat = target["category_id"]
            coco80_cat = coco91_to_coco80_dict[cat]
            cat = np.array([[coco80_cat]])

            x, y, w, h = target["bbox"]
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
            xyxy = np.array([[x1, y1, x2, y2]])
            xywhn = xyxy2xywhn(xyxy, w0, h0)
            labels.append(np.hstack((cat, xywhn)))

        if labels:
            labels = np.vstack(labels)
        else:
            if self.val_mode:
                # for some image without label
                labels = np.zeros((1, 5))
            else:
                raise ValueError(f"set val_mode = False to use images with labels")

        return labels

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return (
            np.concatenate([i[None] for i in im], axis=0),
            np.concatenate(label, 0),
            path,
            shapes,
        )


def create_dataloader(args):
    image_dir_path = os.path.join(args.data_path, "val2017")
    label_json_path = os.path.join(
        args.data_path, "annotations", "instances_val2017.json"
    )
    dataset = CalibrationCOCO(
        image_dir_path=image_dir_path,
        label_json_path=label_json_path,
        image_size=args.img_size,
        stride=32,
        pad_color=114,
    )

    calibration_dataset = dataset

    if args.num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(args.num_samples)
        )

    assert len(dataset), f"data size is 0, check data path please"
    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=len(calibration_dataset) % args.batch_size != 0,
        collate_fn=dataset.collate_fn,
        num_workers=args.workers,
    )
    verify_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=len(dataset) % args.batch_size != 0,
        collate_fn=dataset.collate_fn,
        num_workers=args.workers,
    )

    return calibration_dataloader, verify_dataloader
