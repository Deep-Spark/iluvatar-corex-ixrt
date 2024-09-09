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
from ixrt.deploy.quantizer.quant_paramter import QuantParameter


class QuantParameterTensor(QuantParameter):
    @classmethod
    def from_quant_param(cls, qparam: QuantParameter):
        return QuantParameterTensor(**qparam.to_dict())

    def to(self, device) -> "QuantParameterTensor":
        if device == self._current_device:
            if self._dev_param is None:
                return self
            return self._dev_param

        dev_param = QuantParameterTensor(**self.to_dict())
        dev_param.scale = torch.tensor(dev_param.scale, device=device)
        dev_param.zero_point = torch.tensor(dev_param.zero_point, device=device)
        self._dev_param = dev_param
        self._current_device = device
        return dev_param

    def __post_init__(self):
        super(QuantParameterTensor, self).__post_init__()
        self._dev_param = None
        self._current_device = None
