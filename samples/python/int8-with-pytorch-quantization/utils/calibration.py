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

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T


class CalibrationImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(CalibrationImageNet, self).__init__(*args, **kwargs)
        img2label_path = os.path.join(self.root, "val_map.txt")
        if not os.path.exists(img2label_path):
            raise FileNotFoundError(f"Not found label file `{img2label_path}`.")

        self.img2label_map = self.make_img2label_map(img2label_path)

    def make_img2label_map(self, path):
        with open(path) as f:
            lines = f.readlines()

        img2lable_map = dict()
        for line in lines:
            line = line.lstrip().rstrip().split("\t")
            if len(line) != 2:
                continue
            img_name, label = line
            img_name = img_name.strip()
            if img_name in [None, ""]:
                continue
            label = int(label.strip())
            img2lable_map[img_name] = label
        return img2lable_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target


def create_dataloader(args):
    dataset = CalibrationImageNet(
        args.data_path,
        transform=T.Compose(
            [
                T.Resize(int(256.0 / 224.0 * args.img_size)),
                T.CenterCrop(args.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    calibration_dataset = dataset
    if args.num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(args.num_samples)
        )

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=len(calibration_dataset) % args.batch_size != 0,
        num_workers=args.workers,
    )

    verify_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=len(dataset) % args.batch_size != 0,
        num_workers=args.workers,
    )

    return calibration_dataloader, verify_dataloader
