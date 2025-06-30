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
from ixrt.deploy.utils.seed import manual_seed

manual_seed(43)
device = 0 if torch.cuda.is_available() else "cpu"


class AccuracyAutotuningEvaluator(AutotuningQuantizerEvaluator):
    def __init__(self, dataloader, threshold=None, use_ixrt=True, work_dir="./"):
        super().__init__(optimized_target="max", threshold=threshold)
        self.dataloader = dataloader
        self.use_ixrt = use_ixrt
        self.work_dir = work_dir

    def evaluate_quantized_model(self, executor, model: Graph):
        if self.use_ixrt:
            model_file = os.path.join(self.work_dir, "autotuning_tmp_model.onnx")
            quant_file = os.path.join(self.work_dir, "autotuning_tmp_model-params.json")
            from utils.infer_ixrt import verify_quantized_model

            return infer_by_ixrt(
                self.dataloader,
                model_file,
                quant_file,
                enable=True,
                work_dir=self.work_dir,
            )(model)
        else:
            return compute_model_acc(
                self.dataloader, executor, quant=True, enable=True, return_acc=True
            )(model)[0]


@torch.no_grad()
def quantize_cls_model(args, model, dataloader):
    calibration_dataloader, val_dataloader = dataloader

    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

    executor = create_executor(backend="torch")

    autotuning_quantization(
        model=model,
        calibration_dataloader=calibration_dataloader,
        executor=executor,
        evaluator=AccuracyAutotuningEvaluator(
            val_dataloader,
            threshold=args.threshold,
            use_ixrt=args.use_ixrt,
            work_dir=args.work_dir,
        ),
        validation_dataloader=val_dataloader,
        data_preprocess=lambda x: x[0].to(device),
        automix_precision=args.automix_precision,
        device=device,
        passes=[
            # 计算量化前 FP32 的准确度
            compute_model_acc(
                val_dataloader, executor, quant=False, enable=args.fp32_acc
            ),
            "default",
        ],
    )


def parse_args():
    parser = create_argparser("PTQ Quantization")
    parser.add_argument(
        "--threshold", default=None, type=float, help="autotuning threshold"
    )
    parser.add_argument(
        "--automix_precision",
        action="store_true",
        help="automatically rollback many operators to float type.",
    )
    parser.add_argument("--work_dir", default="./", type=str, help="to save tmp files.")
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
        return verify_quantized_model(args.model, args.quant_params, dataloader[1])

    if args.model.endswith(".onnx"):
        model_name = os.path.basename(args.model)
        model_name = model_name.rsplit(".", maxsplit=1)[0]
        model = args.model
    else:
        model_name = args.model
        model = models.__dict__[model_name](pretrained=True)

        # import timm
        # model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    quantize_cls_model(args, model, dataloader)


if __name__ == "__main__":
    main()
