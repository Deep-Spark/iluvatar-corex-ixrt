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
from tqdm import tqdm

from ixrt.deploy.api import *


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        output = output.argmax(dim=1)
        correct = (output == target).float().sum()
        return correct


def compute_model_acc(
    dataloader,
    executor: TorchExecutor,
    quant=True,
    enable=True,
    device=None,
    return_acc=False,
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

        acc1 = 0.0
        for x, y in tqdm(dataloader, desc="VerifyQuantModel"):
            x = x.to(device)
            y = y.to(device)
            with exec_ctx():
                out = executor.execute_graph(graph, x)
            acc1 += accuracy(out, y, (1,))
        acc1 = acc1 / len(dataloader.dataset)

        if quant:
            print("SimulatedQuantAccuracy:", acc1)
        else:
            print("FP32 Accuracy:", acc1)

        if return_acc:
            return acc1, graph
        else:
            return graph

    return _compute
