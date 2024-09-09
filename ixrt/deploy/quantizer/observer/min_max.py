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


@QUANT_OBSERVERS.registe(alias="minmax")
class MinMaxOberver(BasedRangeObserver):
    def update_tensor_min(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        if per_channel:
            return torch.where(new_tensor < old_tensor, new_tensor, old_tensor)
        else:
            return min(old_tensor, new_tensor)

    def update_tensor_max(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        if per_channel:
            return torch.where(new_tensor > old_tensor, new_tensor, old_tensor)
        else:
            return max(old_tensor, new_tensor)
