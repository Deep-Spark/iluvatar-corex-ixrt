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

from .conv_bn_pass import ConvBnPass
from .convtranpose_add_pass import ConvtranposeAddPass
from .convtranpose_bn_add_pass import ConvtranposeAddBNPass
from .dropout_pass import DropoutPass
from .groupnorm_pass import FuseGroupNormPass
from .hardswish_pass import FuseHardswishPass
from .mish_pass import FuseMishPass
from .silu_pass import FuseSiLUPass
