#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
import torch.utils.data
from pytorch_quantization import calib, enable_onnx_export
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn
from torchvision import models
from tqdm import tqdm
from utils.argparser import create_argparser
from utils.calibration import create_dataloader

quant_modules.initialize()


def parse_args():
    parser = create_argparser("PTQ Quantization")
    args = parser.parse_args()

    args.use_ixquant = not args.use_ixrt

    if args.data_path is None:
        server = "/home/datasets/cv/ImageNet/val"
        if os.path.exists(server):
            args.data_path = server

    return args


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cuda()


if __name__ == "__main__":
    # data
    args = parse_args()
    print(args)
    dataloader, dataloader_test = create_dataloader(args)

    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model = getattr(models, args.model)(pretrained=True).cuda()
    # model = models.vgg16(pretrained=True)
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, dataloader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)

    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
    with enable_onnx_export():
        torch.onnx.export(
            model,
            dummy_input,
            f"quant_{args.model}.onnx",
            verbose=True,
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
