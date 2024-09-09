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

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from ixrt.deploy.ir.data_type import DataType
from ixrt.deploy.ir.operator_type import OperatorType as OP
from ixrt.deploy.quantizer.quant_operator_config import (
    QuantOperatorConfig,
    get_default_quant_operator_config,
)

_DEFAULT_QUANT_DTYPE = [
    DataType.FLOAT,
    DataType.FLOAT16,
    DataType.BFLOAT16,
    DataType.FP8,
    DataType.DOUBLE,
]

_DEFAULT_SHARED_QUANT_PARAMS_TYPES = [
    OP.ADAPTIVE_MAX_POOL,
    OP.EXPAND,
    OP.FLATTEN,
    OP.GATHER,
    OP.GATHER_ELES,
    OP.GATHER_ND,
    OP.GLOBAL_MAX_POOL,
    OP.IDENTITY,
    OP.MAX_POOL,
    OP.MAX_UNPOOL,
    OP.PERMUTE,
    OP.REDUCE_MAX,
    OP.REDUCE_MIN,
    OP.SCATTER,
    OP.SCATTER_ND,
    OP.GATHER_ND,
    OP.SCATTER_ELES,
    OP.SLICE,
    OP.SPLIT,
    OP.SPLIT_TO_SEQUENCE,
    OP.SQUEEZE,
    OP.TOPK,
    OP.TRANSPOSE,
    OP.UNSQUEEZE,
]


@dataclass()
class QuantAnalyzerConfig:
    enable: bool = False
    metric: Union[str, Callable] = "l2"
    reduce_elements: bool = "mean"
    reduce_samples: bool = "mean"
    show_topk: Optional[int] = None
    error_level: str = "layer"


class QuantizerConfig(object):
    def __init__(
        self,
        operator_config: QuantOperatorConfig = None,
        bias_correction: bool = False,
        quant_analyzer: QuantAnalyzerConfig = None,
        quant_dtype: List = None,
        use_qat: bool = False,
        share_quant_params_types: List = None,
    ):
        self.operator_config = operator_config or get_default_quant_operator_config()
        self.bias_correction = bias_correction
        self.quant_analyzer = quant_analyzer or QuantAnalyzerConfig()
        self.quant_dtype = _DEFAULT_QUANT_DTYPE if quant_dtype is None else quant_dtype
        self.use_qat = use_qat
        self.share_quant_params_types = (
            _DEFAULT_SHARED_QUANT_PARAMS_TYPES
            if share_quant_params_types is None
            else share_quant_params_types
        )

    @property
    def use_quant_analyzer(self):
        return self.quant_analyzer.enable

    def enable_quant_analyzer(self):
        self.quant_analyzer.enable = True

    def disable_quant_analyzer(self):
        self.quant_analyzer.enable = False

    @classmethod
    def default_config(cls):
        return cls()

    def copy(self):
        return copy.deepcopy(self)
