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

from .based_range import BasedRangeObserver
from .quant_observer import QUANT_OBSERVERS


@QUANT_OBSERVERS.registe(alias="ema")
class EmaObserver(BasedRangeObserver):
    def __init__(self, *, beta=0.99, **kwargs):
        super(EmaObserver, self).__init__(**kwargs)
        self.beta = beta

    @torch.no_grad()
    def on_watch(self, new_value: torch.Tensor):
        if torch.is_tensor(new_value) and new_value.ndim == 0:
            new_value = new_value.unsqueeze(dim=0)

        if new_value.ndim < 2:
            super(EmaObserver, self).on_watch(new_value)
        for batch in range(new_value.shape[0]):
            super(EmaObserver, self).on_watch(new_value[batch : batch + 1])

    def update_tensor_min(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        return self.beta * old_tensor + (1 - self.beta) * new_tensor

    def update_tensor_max(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        return self.beta * old_tensor + (1 - self.beta) * new_tensor
