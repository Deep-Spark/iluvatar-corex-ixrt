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

# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy

from ...ir.data_type import get_type_bit_width
from ...quantizer.quant_operator_config import QuantMode
from .hist import HistObserver
from .quant_observer import QUANT_OBSERVERS


@QUANT_OBSERVERS.registe(alias="entropy")
class EntropyObserver(HistObserver):
    def __init__(self, *, start_bin=None, stride=1, **kwargs):
        super(EntropyObserver, self).__init__(**kwargs)
        self.start_bin = start_bin
        self.stride = stride

        if self.quant_policy.mode == QuantMode.ASYMMETRICAL:
            raise RuntimeError(
                "The entropy calibration is not support asymmetrical mode."
            )

    def compute_amax_from_hist(self, hist: torch.Tensor, tensor_max: torch.Tensor):
        if (isinstance(hist, int) and hist == 0) or hist.sum() < self.num_bins:
            return tensor_max

        total_data = hist.sum()

        quant_bit_width = get_type_bit_width(self.quant_policy.qtype)
        nbins = 1 << (quant_bit_width - 1)

        start_bin = self.num_bins // 2 if self.start_bin is None else self.start_bin

        bins = hist.detach().cpu().numpy()

        best_bins = self.num_bins
        best_entropy = float("inf")
        new_density_counts = np.zeros(nbins, dtype=np.float64)

        def _normalize_distr(distr):
            summ = np.sum(distr)
            if summ != 0:
                distr = distr / summ

            return distr

        for i in range(start_bin, self.num_bins + 1, self.stride):
            new_density_counts.fill(0)
            space = np.linspace(0, i, num=nbins + 1)
            digitized_space = np.digitize(range(i), space) - 1

            digitized_space[bins[:i] == 0] = -1

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bins[idx]

            counter = Counter(digitized_space)
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            new_density = np.zeros(i, dtype=np.float64)
            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = np.sum(new_density) + np.sum(bins[i:])

            new_density = _normalize_distr(new_density)

            reference_density = np.array(bins[: len(digitized_space)])
            reference_density[-1] += np.sum(bins[i:])

            total_counts_old = np.sum(reference_density)
            if (
                round(total_counts_new) != total_data
                or round(total_counts_old) != total_data
            ):
                # print("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={} on {}".format(
                #     total_counts_new, total_counts_old, total_data, self.watched_variable().name))
                continue

            reference_density = _normalize_distr(reference_density)

            ent = entropy(reference_density, new_density)

            if ent <= best_entropy:
                best_entropy = ent
                best_bins = i

        amax = tensor_max * (best_bins / self.num_bins)
        return amax.detach().cpu()
