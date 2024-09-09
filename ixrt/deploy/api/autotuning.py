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

import os
import warnings
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from ..autotuning import ParameterSpace
from ..autotuning.autotuning_quantization import (
    AutoMixPrecisionParamSpace,
    AutoTuneParameter,
    AutotuningEvaluator,
    AutotuningQuantizer,
    AutotuningQuantizerEvaluator,
    QuantizationErrorEvaluator,
    QuantizationSimilarityEvaluator,
    create_autotuning_observers,
    create_basic_observers,
    create_basic_param_space,
    create_parameter_space,
)
from ..fusion import BasePass, PassSequence, create_passes
from ..ir import BaseExecutor, BaseTarget
from ..quantizer import QuantOperatorObserverConfig
from ..quantizer.observer import QuantVariableObserver
from .executor import create_executor
from .pipeline import Pipeline, ToDevice
from .source import create_source
from .target import create_target

__all__ = [
    "autotuning_quantization",
    "AutotuningQuantizer",
    "AutotuningQuantizerEvaluator",
    "AutoMixPrecisionParamSpace",
    "AutotuningEvaluator",
    "AutoTuneParameter",
    "create_autotuning_observers",
    "create_basic_observers",
    "create_basic_param_space",
    "create_parameter_space",
    "QuantizationErrorEvaluator",
    "QuantizationSimilarityEvaluator",
]


def _create_passes(passes: Union[str, List, BasePass]):
    return PassSequence(*create_passes(passes))


def autotuning_quantization(
    model,
    calibration_dataloader: torch.utils.data.DataLoader,
    executor: BaseExecutor = None,
    evaluator: AutotuningQuantizerEvaluator = None,
    validation_dataloader: torch.utils.data.DataLoader = None,
    data_preprocess: Callable = None,
    export_target: BaseTarget = None,
    param_space: ParameterSpace = None,
    observers: Union[str, Union[List[str], List[QuantVariableObserver]]] = None,
    disable_quant_names: list = None,
    disable_quant_types: list = None,
    operator_config_by_type: Dict[str, QuantOperatorObserverConfig] = None,
    operator_config_by_name: Dict[str, QuantOperatorObserverConfig] = None,
    automix_precision: bool = False,
    device=None,
    passes: Union[str, List, BasePass] = "default",
    save_quant_onnx_path=None,
    save_quant_params_path=None,
    verbose: int = 1,
):
    """
    - 自动选择定点算法：会根据 param_space 定义的参数空间去选择定点算法已经量化配置
    - 自动混合精度：会根据每层的量化误差去自动回退算子到 Float 类型进行计算

    :param model: ONNX 文件路径或者 PyTorch  `nn.Module`
    :param calibration_dataloader: 校准数据集
    :param executor: 执行器
    :param evaluator: 评估器，用于评估量化后的模型质量，建议自定义一个评估值，来计算模型的评价指标，比如 Accuracy, mAP, mIoU 等
        如果为 None，那么就会使用量化前后模型输出的相似度来进行搜索，相当于最大化量化后模型输出的相似度
    :param validation_dataloader: 验证集，如果没有提供，将使用 calibration_dataloader 作为验证集
    :param data_preprocess: 数据预处理，将 Dataloader 中的 Batch 应用 data_preprocess 转换量化工具支持的输入
    :param export_target: 用于保存量化后的模型
    :param param_space: 用于搜索的参数空间，如果默认的搜索空间，那么可以自定义一个参数空间用于搜索
    :param observers: 定点算法，如果 param_space 没有提供，可以使用 observers 参数去提供搜索的定点算法
    :param disable_quant_names: 禁止量化的算子名字
    :param disable_quant_types: 禁止量化的算子类型
    :param operator_config_by_type: 通过类型对算子进行自定义配置
    :param operator_config_by_name: 通过名字对算子进行自定义配置
    :param automix_precision: 是否启用自动混合精度
    :param device: 运行设备号
    :param passes: 量化前对 Graph 进行处理的 Pass
    :param save_quant_onnx_path: 保存量化后的 onnx 到本地的路径
    :param save_quant_params_path: 保存量化参数到本地的路径
    :param verbose: 用于控制中间日志的状态，值为 0 只显示关键状态的日志，为 1 显示量化的配置，为 2 显示算子的误差
    """
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    if data_preprocess is None:
        warnings.warn(
            "autotuning_quantization: `data_preprocess` is none, the default preprocess is used."
        )

        def _default_preprocess(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if torch.is_tensor(x):
                x = x.to(device)

            return x

        data_preprocess = _default_preprocess

    if executor is None:
        executor = create_executor(backend="torch")

    executor.set_device(device)

    example_inputs = next(iter(calibration_dataloader))
    example_inputs = data_preprocess(example_inputs)

    if isinstance(model, str):
        model_name = os.path.basename(model)
        model_name = model_name.rsplit(".", maxsplit=1)[0]
    elif isinstance(model, torch.nn.Module):
        model_name = model.__class__.__name__
    else:
        model_name = "quantized-model"

    if save_quant_onnx_path is None:
        save_quant_onnx_path = f"{model_name}-quant.onnx"

    if save_quant_params_path is None:
        save_quant_params_path = f"{model_name}-quant-params.json"

    if export_target is None:
        print(
            f"Save quantized model to `{save_quant_onnx_path}` and `{save_quant_params_path}`."
        )
        export_target = create_target(
            saved_path=save_quant_onnx_path,
            example_inputs=example_inputs,
            quant_params_path=save_quant_params_path,
            name=model_name,
        )

    quantizer = AutotuningQuantizer(
        calibration_dataloader=calibration_dataloader,
        executor=executor,
        evaluator=evaluator,
        validation_dataloader=validation_dataloader,
        data_preprocess=data_preprocess,
        export_target=export_target,
        param_space=param_space,
        observers=observers,
        disable_quant_names=disable_quant_names,
        disable_quant_types=disable_quant_types,
        operator_config_by_type=operator_config_by_type,
        operator_config_by_name=operator_config_by_name,
        automix_precision=automix_precision,
        verbose=verbose,
    )

    pipeline = Pipeline(
        create_source(model, example_inputs=example_inputs),
        ToDevice(device=device),
        _create_passes(passes),
        quantizer,
    )

    graph = pipeline.run()

    best_params = quantizer.get_best_params()
    if best_params is not None:
        print(f"Autotuning best params: {best_params.best_params}")
        print(f"Autotuning best result: {best_params.best_result}")

    return graph
