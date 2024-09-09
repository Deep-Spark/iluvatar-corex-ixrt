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

from ..fusion.base_pass import BasePass, PassSequence
from ..fusion.factory import *
from ..ir import (
    DataType,
    ExecutorHook,
    Graph,
    GraphTransform,
    LambdaExecutorHook,
    Operator,
    OperatorType,
    Placeholder,
    Variable,
    VariableOptions,
    VariableType,
)
from ..quantizer import (
    PostTrainingStaticQuantizer,
    QuantGrain,
    QuantizerConfig,
    QuantMode,
    QuantOperatorObserverConfig,
    QuantPolicy,
    get_default_quant_operator_config,
)
from ..quantizer.observer import create_observer
from .autotuning import *
from .executor import TorchExecutor, create_executor
from .operator import BaseOperator, registe_operator
from .pipeline import Pipeline, ToDevice
from .quant_params import LoadQuantParamtersPPQStype, SaveQuantParameterPPQStyle
from .quantization import static_quantize, verify_quantized_model
from .source import create_source
from .target import create_target
