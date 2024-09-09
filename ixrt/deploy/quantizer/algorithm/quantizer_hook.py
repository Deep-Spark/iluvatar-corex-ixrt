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

from ixrt.deploy.core.hook import BaseHook


class QuantizerHook(BaseHook):
    def set_quantizer(self, quantizer):
        self.quantizer = quantizer

    def on_quantize_start(self):
        pass

    def on_quantize_end(self):
        pass

    def on_init_start(self):
        pass

    def on_init_end(self):
        pass

    def on_finetune_start(self):
        pass

    def on_finetune_end(self):
        pass

    def on_convert_start(self):
        pass

    def on_convert_end(self):
        pass
