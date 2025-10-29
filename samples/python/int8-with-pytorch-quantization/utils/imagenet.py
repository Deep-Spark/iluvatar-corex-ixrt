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

import torch
import torchvision

from . import presets_classification as presets


def get_datasets(
    traindir,
    resize_size=256,
    crop_size=224,
    auto_augment_policy=None,
    random_erase_prob=0.0,
):
    # Data loading code
    print("Loading data")
    print("Loading training data")
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=crop_size,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    return dataset


def get_input_size(model):
    biger_input_size_models = ["inception"]
    resize_size = 256
    crop_size = 224
    for bi_model in biger_input_size_models:
        if bi_model in model:
            resize_size = 342
            crop_size = 299

    return resize_size, crop_size


def load_data(train_dir, args):
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    if args.img_size is None:
        resize_size, crop_size = get_input_size(args.model)
    else:
        resize_size, crop_size = args.img_size, args.img_size
    dataset = get_datasets(
        train_dir,
        auto_augment_policy=auto_augment_policy,
        random_erase_prob=random_erase_prob,
        resize_size=resize_size,
        crop_size=crop_size,
    )
    distributed = getattr(args, "distributed", False)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    return dataset, train_sampler


def _create_torch_dataloader(train_dir, args):
    dataset, train_sampler = load_data(train_dir, args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=len(dataset) % args.batch_size != 0,
        num_workers=args.workers,
        pin_memory=True,
    )

    return data_loader


def create_train_dataloader(train_dir, args):
    print("Creating data loaders")
    return _create_torch_dataloader(train_dir, args)
