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

from abc import abstractmethod

import torch

from .. import quant_function
from ..quant_operator_config import QuantGrain, QuantMode, QuantPolicy
from ..quant_paramter import QuantParameter
from ..utils import tensor_to_list
from .quant_observer import QuantVariableObserver


class BasedRangeObserver(QuantVariableObserver):
    def __init__(self, quant_policy: QuantPolicy):
        super(BasedRangeObserver, self).__init__(quant_policy)

        self.tensor_min: torch.Tensor = None
        self.tensor_max: torch.Tensor = None

        if quant_policy.grain == QuantGrain.PER_CHANNEL:
            if quant_policy.quant_dim is None:
                raise RuntimeError(
                    "The quantized dim of policy must be given, when grain is per channel."
                )
            if quant_policy.quant_dim < 0:
                raise RuntimeError(
                    f"Invalid quantized dim, got {quant_policy.quant_dim}."
                )

    @torch.no_grad()
    def on_watch(self, new_value: torch.Tensor):
        if not torch.is_tensor(new_value):
            raise RuntimeError(
                "The watched value is not torch.Tensor, "
                f"got {new_value}, "
                f"please revoke the observer on `{self.watched_variable()}`."
            )

        if self.quant_policy.grain == QuantGrain.PER_TENSOR:
            tensor_min = new_value.min()
            tensor_max = new_value.max()
            if self.tensor_min is not None:
                tensor_min = self.update_tensor_min(tensor_min, self.tensor_min, False)
                tensor_max = self.update_tensor_max(tensor_max, self.tensor_max, False)
            self.tensor_min = tensor_min
            self.tensor_max = tensor_max

        elif self.quant_policy.grain == QuantGrain.PER_CHANNEL:
            reduce_dims = list(range(new_value.ndim))
            if self.quant_policy.quant_dim is not None:
                reduce_dims.remove(self.quant_policy.quant_dim)

            tensor_min = torch.amin(new_value, dim=reduce_dims)
            tensor_max = torch.amax(new_value, dim=reduce_dims)
            if self.tensor_min is not None:
                tensor_min = self.update_tensor_min(tensor_min, self.tensor_min, True)
                tensor_max = self.update_tensor_max(tensor_max, self.tensor_max, True)
            self.tensor_min = tensor_min
            self.tensor_max = tensor_max

    @abstractmethod
    def update_tensor_min(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def update_tensor_max(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, per_channel: bool
    ) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def get_quant_parameters(self) -> QuantParameter:
        if self.tensor_min is None or self.tensor_max is None:
            raise RuntimeError(
                "Don't watch any new value, "
                f"please check the value of `{self.watched_variable()}`."
            )

        scale, zero_point = quant_function.compute_scale_zero_point(
            self.tensor_min, self.tensor_max, self.quant_policy
        )

        return QuantParameter(
            per_channel=self.quant_policy.grain == QuantGrain.PER_CHANNEL,
            scale=tensor_to_list(scale, to_scaler=True),
            zero_point=tensor_to_list(zero_point, to_scaler=True),
            tensor_min=self._tensor_to_value(self.tensor_min),
            tensor_max=self._tensor_to_value(self.tensor_max),
            qtype=self.quant_policy.qtype.name.lower(),
            qtype_min=self.quant_policy.qtype_min,
            qtype_max=self.quant_policy.qtype_max,
            quant_dim=self.quant_policy.quant_dim,
            symmetrical=self.quant_policy.mode == QuantMode.SYMMETRICAL,
        )

    def _tensor_to_value(self, tensor: torch.Tensor):
        if tensor.numel() == 1:
            return tensor.reshape(-1).cpu().item()
        return tensor.cpu().tolist()
