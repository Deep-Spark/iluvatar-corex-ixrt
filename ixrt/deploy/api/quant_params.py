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

from ixrt.deploy.fusion import BasePass
from ixrt.deploy.quantizer.save_quant_param import SaveQuantParameterPPQStyle


class LoadQuantParamtersPPQStype(BasePass):
    def __init__(self, path):
        self.path = path

    def process(self, graph):
        return SaveQuantParameterPPQStyle.load(graph, self.path)
