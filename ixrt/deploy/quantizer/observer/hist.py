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

import abc
from typing import List, Union

import torch

from ..quant_operator_config import QuantGrain, QuantMode
from ..quant_paramter import QuantParameter
from .min_max import MinMaxOberver


class HistObserver(MinMaxOberver):
    def __init__(self, num_bins=2048, **kwargs):
        super(HistObserver, self).__init__(**kwargs)

        self.num_bins = num_bins
        self.hist: Union[torch.Tensor, List[torch.Tensor]] = (
            0 if self.quant_policy.grain == QuantGrain.PER_TENSOR else []
        )
        self.phase = "FindMinMax"

    def start_find_minmax(self):
        self.phase = "FindMinMax"

    def finished_find_minmax(self):
        self.phase = "GenerateHistogram"

    @torch.no_grad()
    def on_watch(self, new_value: torch.Tensor):
        if self.phase == "FindMinMax":
            return super(HistObserver, self).on_watch(new_value)

        if self.quant_policy.mode == QuantMode.SYMMETRICAL:
            new_value = new_value.abs()

        if self.quant_policy.grain == QuantGrain.PER_CHANNEL:
            qdim = self.quant_policy.quant_dim
            if len(self.hist) == 0:
                self.hist = [0] * new_value.shape[qdim]

            for qdim_idx in range(new_value.shape[qdim]):
                if self.quant_policy.mode == QuantMode.SYMMETRICAL:
                    hist_max = max(
                        self.tensor_min[qdim_idx].abs(), self.tensor_max[qdim_idx].abs()
                    )
                    hist = torch.histc(
                        torch.select(new_value, qdim, qdim_idx),
                        self.num_bins,
                        0,
                        hist_max,
                    )
                else:
                    hist = torch.histc(
                        torch.select(new_value, qdim, qdim_idx),
                        self.num_bins,
                        self.tensor_min[qdim_idx],
                        self.tensor_max[qdim_idx],
                    )
                self.hist[qdim_idx] += hist.int()
        else:
            if self.quant_policy.mode == QuantMode.SYMMETRICAL:
                hist_max = max(self.tensor_min.abs(), self.tensor_max.abs())
                self.hist += torch.histc(new_value, self.num_bins, 0, hist_max).int()
            else:
                self.hist += torch.histc(
                    new_value, self.num_bins, self.tensor_min, self.tensor_max
                ).int()

    def get_quant_parameters(self) -> QuantParameter:
        self.compute_quant_params_from_hist()
        return super(HistObserver, self).get_quant_parameters()

    @torch.no_grad()
    def compute_quant_params_from_hist(self):
        if self.quant_policy.grain == QuantGrain.PER_TENSOR:
            new_tensor_max = self.compute_amax_from_hist(
                self.hist, max(self.tensor_min.abs(), self.tensor_max.abs())
            )
            self.tensor_min = new_tensor_max
            self.tensor_max = new_tensor_max
        else:
            new_tenosr_max = []
            for channel_idx, hist in enumerate(self.hist):
                tenosr_max = self.compute_amax_from_hist(
                    hist,
                    max(
                        self.tensor_min[channel_idx].abs(),
                        self.tensor_max[channel_idx].abs(),
                    ),
                )
                new_tenosr_max.append(tenosr_max)
            new_tenosr_max = torch.stack(new_tenosr_max)
            self.tensor_min = new_tenosr_max
            self.tensor_max = new_tenosr_max

    @abc.abstractmethod
    def compute_amax_from_hist(self, hist: torch.Tensor, tensor_max: torch.Tensor):
        raise NotImplementedError()
