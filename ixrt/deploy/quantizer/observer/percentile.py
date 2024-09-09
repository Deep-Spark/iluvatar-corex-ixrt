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

from ..quant_operator_config import QuantGrain, QuantMode
from ..quant_paramter import QuantParameter
from .based_range import BasedRangeObserver
from .hist import HistObserver
from .quant_observer import QUANT_OBSERVERS


@QUANT_OBSERVERS.registe(alias="percentile")
class PercentileObserver(BasedRangeObserver):
    def __init__(self, percentile=99.99, **kwargs):
        super(PercentileObserver, self).__init__(**kwargs)
        self.percentile = percentile
        self.tensor_min_tile = []
        self.tensor_max_tile = []

        if self.quant_policy.mode == QuantMode.ASYMMETRICAL:
            raise RuntimeError("Not support asymmetrical mode")

    @torch.no_grad()
    def on_watch(self, new_value: torch.Tensor):
        if not torch.is_tensor(new_value):
            raise RuntimeError(
                "The watched value is not torch.Tensor, "
                f"got {new_value}, "
                f"please revoke the observer on `{self.watched_variable()}`."
            )

        if new_value.numel() == 0:
            return

        new_value = new_value.abs()

        if self.quant_policy.grain == QuantGrain.PER_TENSOR:
            new_value = new_value.reshape(-1)
            kth = int(new_value.numel() * (self.percentile * 0.01))
            kth = min(kth, new_value.numel())
            kth = max(kth, 1)
            self.tensor_max_tile.append(
                torch.kthvalue(new_value, kth).values.detach().cpu()
            )

        else:
            qdim = self.quant_policy.quant_dim
            tiles = []
            for qdim_idx in range(new_value.shape[qdim]):
                layer = torch.select(new_value, qdim, qdim_idx).reshape(-1)
                kth = int(layer.numel() * (self.percentile * 0.01))
                kth = min(kth, layer.numel())
                kth = max(kth, 1)
                tiles.append(torch.kthvalue(layer, kth).values)
            tiles = torch.stack(tiles).detach().cpu()
            self.tensor_max_tile.append(tiles)

    def get_quant_parameters(self) -> QuantParameter:
        if len(self.tensor_max_tile) == 0:
            device = "cpu"
            if torch.cuda.is_available():
                device = torch.cuda.current_device()

            self.tensor_max = torch.tensor(0.0, dtype=torch.float32, device=device)
            self.tensor_min = self.tensor_max
        else:
            max_tiles = torch.stack(self.tensor_max_tile)
            self.tensor_max = max_tiles.mean(dim=0)
            self.tensor_min = self.tensor_max

        return super(PercentileObserver, self).get_quant_parameters()


@QUANT_OBSERVERS.registe(alias="hist_percentile")
class BasedHistPercentileObserver(HistObserver):
    def __init__(self, percentile=99.99, **kwargs):
        super(BasedHistPercentileObserver, self).__init__(**kwargs)
        self.percentile = percentile / 100.0

    def compute_quant_params_from_hist(self):
        if self.quant_policy.mode == QuantMode.SYMMETRICAL:
            return super(
                BasedHistPercentileObserver, self
            ).compute_quant_params_from_hist()

        if self.quant_policy.grain == QuantGrain.PER_TENSOR:
            new_tensor_max = self.compute_amax_from_hist(self.hist, self.tensor_max)
            new_tensor_min = self.compute_amin_from_hist(self.hist, self.tensor_min)

            self.tensor_min = new_tensor_min
            self.tensor_max = new_tensor_max
        else:
            new_tenosr_max = []
            new_tenosr_min = []
            for channel_idx, hist in enumerate(self.hist):
                tenosr_max = self.compute_amax_from_hist(
                    hist, self.tensor_max[channel_idx]
                )
                new_tenosr_max.append(tenosr_max)

                tensor_min = self.compute_amin_from_hist(
                    hist, self.tensor_min[channel_idx]
                )
                new_tenosr_min.append(tensor_min)

            self.tensor_min = torch.stack(new_tenosr_max)
            self.tensor_max = torch.stack(new_tenosr_min)

    def compute_amin_from_hist(self, hist: torch.Tensor, tensor_min: torch.Tensor):
        hist_r = reversed(hist)
        new_tensor_min = self.compute_amax_from_hist(hist_r, tensor_min)
        if new_tensor_min > 0:
            new_tensor_min = tensor_min + (tensor_min - new_tensor_min)

        return new_tensor_min

    def compute_amax_from_hist(self, hist: torch.Tensor, tensor_max: torch.Tensor):
        if isinstance(hist, int) and hist == 0:
            return tensor_max

        total = hist.sum()
        perentile_count = int(total * self.percentile)
        hist_spaces = hist.cumsum(dim=0)
        bin_idx = len(hist_spaces)
        for i, count in enumerate(reversed(hist_spaces)):
            if count <= perentile_count:
                bin_idx = i
                break

        return tensor_max * max(1 - bin_idx / len(hist), 0.5)
