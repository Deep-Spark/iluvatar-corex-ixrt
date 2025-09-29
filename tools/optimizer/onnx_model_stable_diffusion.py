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

import onnx
from passes.fusion_biasgelu import FusionBiasGelu
from passes.fusion_customfc import (
    FusionCustomFC,
    FusionCustomFCActivation,
)
from passes.fusion_fastgelu import FusionFastGelu
from passes.fusion_gelu import FusionGelu
from passes.fusion_options import FusionOptions
from passes.fusion_qordered_gelu import FusionQOrderedGelu
from passes.fusion_reshape import FusionReshape
from passes.fusion_shape import FusionShape
from passes.fusion_skiplayernorm import (
    FusionBiasSkipLayerNormalization,
    FusionSkipLayerNormalization,
)
from passes.fusion_stable_attention import FusionStableAttention
from passes.fusion_cross_attention import FusionCrossAttention
from passes.fusion_utils import FusionUtils

from passes.onnx_model import OnnxModel

logger = getLogger(__name__)


class StableDiffusionOnnxModel(OnnxModel):
    def __init__(self, model: onnx.ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BERT ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (
            num_heads > 0 and hidden_size % num_heads == 0
        )

        super().__init__(model)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.utils = FusionUtils(self)

    def fuse_attention(self):
        fusion = FusionStableAttention(self,self.hidden_size, self.num_heads)
        fusion.apply()

        fusion = FusionCrossAttention(self,self.hidden_size, self.num_heads)
        fusion.apply()

    def fuse_custom_fc(self):
        fusion = FusionCustomFC(self)
        fusion.apply()

    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()
        fusion = FusionFastGelu(self)
        fusion.apply()
        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedGelu(self)
        fusion.apply()

    def fuse_bias_gelu(self, is_fastgelu):
        fusion = FusionBiasGelu(self, is_fastgelu)
        fusion.apply()

    def fuse_add_bias_skip_layer_norm(self):
        fusion = FusionBiasSkipLayerNormalization(self)
        fusion.apply()

    def fuse_reshape(self):
        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_shape(self):
        fusion = FusionShape(self)
        fusion.apply()

    def fuse_skip_layer_norm(self):
        fusion = FusionSkipLayerNormalization(self)
        fusion.apply()

    def fuse_custom_fc_activation(self):
        fusion = FusionCustomFCActivation(self)
        fusion.apply()

    def optimize(
        self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False
    ):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.fuse_reshape()

        # if (options is None) or options.enable_skip_layer_norm:
        #     self.fuse_skip_layer_norm()

        self.fuse_attention()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        self.fuse_custom_fc()

        
        self.fuse_shape()


        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()


        # Bias fusion is done after postprocess to avoid extra Reshape between bias and Gelu/FastGelu/SkipLayerNormalization
        if (options is None) or options.enable_bias_gelu:
            # Fuse Gelu and Add Bias before it.
            self.fuse_bias_gelu(is_fastgelu=True)
            self.fuse_bias_gelu(is_fastgelu=False)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        self.fuse_custom_fc_activation()
        
        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        if add_dynamic_axes:
            self.use_dynamic_axes()

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "Attention",
            "Gelu",
            "SkipLayerNormalization",
            "QOrderedMatMul",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)
        logger.info(f"Optimized operators:{op_count}")
        return op_count

    def is_fully_optimized(self):
        """
        Returns True when the model is fully optimized.
        """
        op_count = self.get_fused_operator_statistics()
        attention = op_count["Attention"] 
        gelu = op_count["Gelu"] + op_count["BiasGelu"] + op_count["FastGelu"]
        layer_norm = op_count["SkipLayerNormalization"]
        is_perfect = (
            (attention > 0)
            and (attention == gelu)
            and (layer_norm >= 2 * attention)
        )
        return is_perfect
