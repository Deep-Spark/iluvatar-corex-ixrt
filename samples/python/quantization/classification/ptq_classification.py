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
from torchvision import models
from utils.accuracy import compute_model_acc
from utils.argparser import create_argparser
from utils.calibration import create_dataloader
from utils.infer_ixrt import infer_by_ixrt, verify_quantized_model

from ixrt.deploy.api import *
from ixrt.deploy.ir.operator_type import OperatorType
from ixrt.deploy.utils.seed import manual_seed

manual_seed(43)
device = 0 if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def quantize_cls_model(args, model_name, model, dataloader):
    calibration_dataloader, verify_dataloader = dataloader

    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

    executor = create_executor(backend="torch")

    qconfig = QuantizerConfig(
        operator_config=get_default_quant_operator_config(
            activation_observer=args.observer
        )
    )
    qconfig.bias_correction = not args.disable_bias_correction
    qconfig.quant_analyzer.enable = args.analyze

    test_inputs = next(iter(verify_dataloader))[0].to(device)

    quant_pipeline = Pipeline(
        create_source(
            model,
            example_inputs=test_inputs,
        ),
        ToDevice(device=device),
        compute_model_acc(
            verify_dataloader, executor, quant=False, enable=args.fp32_acc
        ),
        get_default_passes(),
        PostTrainingStaticQuantizer(
            calibration_dataloader,
            executor=executor,
            qconfig=qconfig,
            preprocess=lambda x: x[0].to(device),
            val_dataloader=verify_dataloader,
        ),
        compute_model_acc(
            verify_dataloader, executor, quant=True, enable=args.use_ixquant
        ),
        create_target(
            saved_path=f"{model_name}-quant.onnx",
            example_inputs=test_inputs,
            quant_params_path=f"./{model_name}-params.json",
            name=model_name,
        ),
        infer_by_ixrt(
            model_path=f"{model_name}-quant.onnx",
            quant_params_path=f"./{model_name}-params.json",
            dataloader=verify_dataloader,
            enable=args.use_ixrt,
        ),
    )
    quant_pipeline.run()


def parse_args():
    parser = create_argparser("PTQ Quantization")
    args = parser.parse_args()

    args.use_ixquant = not args.use_ixrt

    if args.data_path is None:
        server = "/home/datasets/cv/ImageNet/val"
        if os.path.exists(server):
            args.data_path = server

    return args


def main():
    args = parse_args()
    print(args)

    dataloader = create_dataloader(args)

    if args.quant_params is not None:
        return verify_quantized_model(args, dataloader[1])

    if args.model.endswith(".onnx"):
        model_name = os.path.basename(args.model)
        model_name = model_name.rsplit(".", maxsplit=1)[0]
        model = args.model
    else:
        model_name = args.model
        model = models.__dict__[model_name](pretrained=True)

        # import timm
        # model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    quantize_cls_model(args, model_name, model, dataloader)


if __name__ == "__main__":
    main()
