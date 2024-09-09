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

from typing import List

import numpy as np
import torch
from ixrt.deploy.core import Registry

ANALYZER_METRICS = Registry("AnalyzerMetricRegistry")


def get_metric(name, **kwargs):
    return ANALYZER_METRICS.get(name, **kwargs)


def reshape_to_two_dim(*arrs):
    arrs: List[np.ndarray] = list(arrs)
    for i, arr in enumerate(arrs):
        if arr.ndim == 1:
            arrs[i] = arr.reshape(-1, 1)
        else:
            arrs[i] = arr.reshape(arr.shape[0], -1)
    return arrs


def batch_reduce(v: torch.Tensor, reduction: str):
    reduction = reduction.lower()
    v: torch.Tensor = v.reshape(v.shape[0], -1).float()
    if reduction == "sum":
        return v.sum()
    if reduction == "mean":
        return v.mean(dim=1)
    if reduction == "max":
        return v.max(dim=1).values
    if reduction == "min":
        return v.min(dim=1).values
    if reduction == "none":
        return v


@ANALYZER_METRICS.registe(alias="l1")
def l1_norm(real_out: torch.Tensor, quant_out: torch.Tensor, reduction: str):
    real_out, quant_out = reshape_to_two_dim(real_out, quant_out)
    error = (real_out - quant_out).abs()
    return batch_reduce(error, reduction)


@ANALYZER_METRICS.registe(alias="l2")
def l2_norm(real_out: torch.Tensor, quant_out: torch.Tensor, reduction: str):
    real_out, quant_out = reshape_to_two_dim(real_out, quant_out)
    error = (real_out - quant_out).pow(2)
    return batch_reduce(error, reduction)


@ANALYZER_METRICS.registe(alias="similarity")
def cosine_similarity(real_out: torch.Tensor, quant_out: torch.Tensor, reduction: str):
    real_out, quant_out = reshape_to_two_dim(real_out, quant_out)
    similarity = torch.cosine_similarity(real_out, quant_out, dim=1)
    return batch_reduce(similarity, reduction)


@ANALYZER_METRICS.registe(alias="relative")
def relative_l1(real_out: torch.Tensor, quant_out: torch.Tensor, reduction: str):
    real_out, quant_out = reshape_to_two_dim(real_out, quant_out)
    error = (real_out - quant_out).abs() / real_out.abs().max()
    return batch_reduce(error, reduction)
