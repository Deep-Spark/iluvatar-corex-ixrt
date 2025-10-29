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

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional

from onnx import ModelProto
from passes.fusion_options import FusionOptions
from passes.fusion_utils import FusionUtils
from passes.onnx_model import OnnxModel
from passes.fuse_lstm import FusionLstmSqueeze
from passes.fuse_lstm import FusionLstmTranspose

logger = getLogger(__name__)


class RnnOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize Rnn ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (
            num_heads > 0 and hidden_size % num_heads == 0
        )
        super().__init__(model)
        self.utils = FusionUtils(self)

    def fuse_lstm_squeeze(self):
        fusion = FusionLstmSqueeze(self)
        fusion.apply()

    def fuse_lstm_transpsose(self):
        fusion = FusionLstmTranspose(self)
        fusion.apply()



    def optimize(
        self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False
    ):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()
        self.fuse_lstm_squeeze()
        self.fuse_lstm_transpsose()
        self.remove_unused_constant()
        logger.info(f"opset version: {self.get_opset_version()}")
