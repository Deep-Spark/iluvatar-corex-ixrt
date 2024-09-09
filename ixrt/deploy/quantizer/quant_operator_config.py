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
from enum import Enum
from typing import Dict, List, Optional, Union

from ..ir.data_type import DataType
from ..ir.operator_type import OperatorType
from .observer.quant_observer import (
    IDENTITY_OBSERVER,
    QuantVariableObserver,
    create_observer,
)

__all__ = [
    "QuantMode",
    "QuantGrain",
    "QuantPolicy",
    "QuantOperatorObserverConfig",
    "QuantOperatorConfig",
    "get_default_quant_operator_config",
    "create_operator_config",
]


INF = float("inf")


class QuantMode(Enum):
    SYMMETRICAL = "Symmetrical"
    ASYMMETRICAL = "Asymmetrical"


class QuantGrain(Enum):
    PER_TENSOR = "PerTensor"
    PER_CHANNEL = "PerChannel"


@dataclass()
class QuantPolicy(object):
    grain: QuantGrain
    mode: QuantMode = QuantMode.SYMMETRICAL

    qtype: DataType = DataType.INT8
    qtype_min: Union[int, float] = None
    qtype_max: Union[int, float] = None

    # if per_chanel is True, quant_dim will be uesed to reduce tensor in special dims.
    # Example:
    #   Def: quant_dim == 1, x = torch.randn(2, 4, 6)
    #   Out: y = torch.amax(x, dim=[0, 2]) # shape: [4]
    quant_dim: Optional[int] = -1

    def __post_init__(self):
        if isinstance(self.grain, str):
            if self.grain.lower() in ["per_channel", "perchannel", "channel"]:
                self.grain = QuantGrain.PER_CHANNEL
            elif self.grain.lower() in ["per_tensor", "perttensor", "tensor"]:
                self.grain = QuantGrain.PER_TENSOR
            else:
                raise RuntimeError(
                    f"Invalid grain, got {self.grain},"
                    f"expect `per_channel` or `per_tensor.`"
                )

        if isinstance(self.mode, str):
            if self.mode.lower() in ["symmetrical", "sym"]:
                self.mode = QuantMode.SYMMETRICAL
            elif self.mode.lower() in ["asymmetrical", "asym"]:
                self.mode = QuantMode.ASYMMETRICAL
            else:
                raise RuntimeError(
                    f"Invalid mode, got {self.mode}, "
                    f"expecte `symmetrical` or `asymmetrical`."
                )

        if isinstance(self.qtype, str):
            self.qtype = DataType.from_name(self.qtype)
        elif isinstance(self.qtype, int):
            self.qtype = DataType.from_value(self.qtype)
        elif not isinstance(self.qtype, DataType):
            raise RuntimeError(f"Invalid quantization type, got {self.qtype}.")

        if self.grain == QuantGrain.PER_CHANNEL and self.quant_dim is None:
            raise RuntimeError(
                "If using per channel grain, "
                "the argument `quant_dim` should be given."
            )

        self._init_qtype()

    def _init_qtype(self):
        # TODO: Add more type
        if self.qtype_min is None or self.qtype_max is None:
            if self.qtype == DataType.INT8:
                self.qtype_min = -127
                self.qtype_max = 127
            elif self.qtype == DataType.UINT8:
                self.qtype_min = 0
                self.qtype_max = 255
            else:
                raise RuntimeError("The range of qtype should be given.")


@dataclass()
class QuantOperatorObserverConfig(object):
    activation: QuantVariableObserver = IDENTITY_OBSERVER
    weight: QuantVariableObserver = IDENTITY_OBSERVER
    bias: QuantVariableObserver = IDENTITY_OBSERVER

    def __post_init__(self):
        self.activation.mark_as_activation()
        self.weight.mark_as_weight()
        self.bias.mark_as_bias()

    def copy(self):
        activation = (
            self.activation if self.activation is None else self.activation.copy()
        )
        weight = self.weight if self.weight is None else self.weight.copy()
        bias = self.bias if self.bias is None else self.bias.copy()
        return QuantOperatorObserverConfig(
            activation=activation, weight=weight, bias=bias
        )

    @classmethod
    def identity_config(cls):
        return cls()

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict=None):
        return self.copy()


class QuantOperatorConfig(object):
    def __init__(
        self,
        global_config: QuantOperatorObserverConfig = None,
        op_configs: Dict[str, QuantOperatorObserverConfig] = None,
        op_name_configs: Dict[str, QuantOperatorObserverConfig] = None,
    ):
        self.global_config = global_config
        if op_configs is None:
            op_configs = dict()
        self.op_configs = op_configs
        if op_name_configs is None:
            op_name_configs = dict()
        self.op_name_configs = op_name_configs

    def set_config_with_op(self, op_type: str, observer: QuantOperatorObserverConfig):
        self.op_configs[op_type] = observer
        return self

    def set_config_with_op_name(self, name: str, observer: QuantOperatorObserverConfig):
        self.op_name_configs[name] = observer
        return self

    def disable_quantize_with_op(self, op_type: Union[str, List]):
        if isinstance(op_type, str):
            op_type = [op_type]

        for _type in op_type:
            self.set_config_with_op(_type, None)

    def disable_quantize_with_op_name(self, name: Union[str, List]):
        if isinstance(name, str):
            name = [name]

        for _name in name:
            self.set_config_with_op_name(_name, None)

    def is_disable_operator(self, operator):
        if (
            operator.name in self.op_name_configs
            and self.op_name_configs[operator.name] is None
        ):
            return True

        if (
            operator.op_type in self.op_configs
            and self.op_configs[operator.op_type] is None
        ):
            return True

        return False

    def set_global_config(self, global_config):
        self.global_config = global_config

    def get_config_with_op(self, op_type: str) -> Optional[QuantOperatorObserverConfig]:
        config = self.op_configs.get(op_type, None)
        return self._postprocess_config(config)

    def get_config_with_op_name(self, op_name) -> Optional[QuantOperatorObserverConfig]:
        config = self.op_name_configs.get(op_name, None)
        return self._postprocess_config(config)

    def get_global_config(self) -> Optional[QuantOperatorObserverConfig]:
        config = self.global_config
        return self._postprocess_config(config)

    def _postprocess_config(self, config: QuantOperatorObserverConfig):
        if config is None:
            return config
        return config.copy()


def get_default_quant_operator_config(
    activation_observer: str = None,
    weight_observer: str = None,
    bias_observer: str = None,
):
    per_tensor_policy = QuantPolicy(QuantGrain.PER_TENSOR)
    per_channel_policy = QuantPolicy(QuantGrain.PER_CHANNEL, quant_dim=0)

    if activation_observer is None:
        activation_observer = "hist_percentile"
    if isinstance(activation_observer, str):
        activation_observer = create_observer(
            activation_observer, quant_policy=per_tensor_policy
        )

    if weight_observer is None:
        weight_observer = "minmax"
    if isinstance(weight_observer, str):
        weight_observer = create_observer(
            weight_observer, quant_policy=per_channel_policy
        )

    if bias_observer is None:
        bias_observer = "identity"
    if isinstance(bias_observer, str):
        bias_observer = create_observer(bias_observer, quant_policy=per_tensor_policy)

    return QuantOperatorConfig(
        global_config=QuantOperatorObserverConfig(
            activation=activation_observer, weight=weight_observer, bias=bias_observer
        ),
        op_configs={
            OperatorType.CONV_TRANSPOSE: QuantOperatorObserverConfig(
                activation=activation_observer,
                weight=create_observer(
                    "minmax",
                    quant_policy=QuantPolicy(QuantGrain.PER_CHANNEL, quant_dim=1),
                ),
                bias=bias_observer,
            )
        },
    )


def create_operator_config(
    observer, operator_config_by_type, operator_config_by_name, weight_observer=None
):
    if isinstance(observer, dict):
        if "quant_policy" in observer:
            quant_policy = observer["quant_policy"]
        else:
            quant_policy = QuantPolicy("per_tensor")
        observer = create_observer(quant_policy=quant_policy, **observer)

    operator_config = get_default_quant_operator_config(
        activation_observer=observer, weight_observer=weight_observer
    )  # type:ignore
    if operator_config_by_type is not None:
        if not isinstance(operator_config_by_type, dict):
            raise RuntimeError(
                f"Invalid `operator_config_by_type`, got {operator_config_by_type}."
            )
        for op_type, op_config in operator_config_by_type.items():
            if not isinstance(op_config, QuantOperatorObserverConfig):
                raise RuntimeError(
                    f"Invalid operator config, op_type: {op_type}, config: {op_config}."
                )
            operator_config.set_config_with_op(op_type, op_config)

    if operator_config_by_name is not None:
        if not isinstance(operator_config_by_name, dict):
            raise RuntimeError(
                f"Invalid `operator_config_by_type`, got {operator_config_by_name}."
            )
        for op_name, op_config in operator_config_by_name.items():
            if not isinstance(op_config, QuantOperatorObserverConfig):
                raise RuntimeError(
                    f"Invalid operator config, op_name: {op_name}, config: {op_config}."
                )
            operator_config.set_config_with_op_name(op_name, op_config)

    return operator_config
