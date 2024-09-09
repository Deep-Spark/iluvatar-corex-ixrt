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
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..backend.torch import TorchExecutor
from ..fusion import BasePass, PassSequence
from ..fusion.factory import create_passes
from ..quantizer import (
    AddQdqPair,
    PostTrainingStaticQuantizer,
    QuantizerConfig,
    QuantOperatorObserverConfig,
    create_operator_config,
    get_default_quant_operator_config,
)
from .executor import create_executor
from .pipeline import Pipeline, ToDevice
from .quant_params import LoadQuantParamtersPPQStype
from .source import create_source
from .target import create_target


def _create_passes(passes: Union[str, List, BasePass]):
    return PassSequence(*create_passes(passes))


def compute_model_acc(
    verification,
    executor: TorchExecutor,
    quant=True,
    enable=True,
    device=None,
):
    if quant:
        exec_ctx = executor.enable_quant_context
    else:
        exec_ctx = executor.disable_quant_context

    if device is None:
        device = executor.default_device()

    def _compute(graph):
        if not enable:
            return graph

        verification.init()
        for data in verification.get_data():
            x = data["x"]
            x = x.to(device)
            with exec_ctx():
                out = executor.execute_graph(graph, x)
            verification.calculate(out, data)
        if quant:
            print("SimulatedQuantAccuracy:", verification.summary())
        else:
            print("FP32 Accuracy:", verification.summary())
        return graph

    return _compute


@torch.no_grad()
def static_quantize(
    model: [str, torch.nn.Module],
    calibration_dataloader: torch.utils.data.DataLoader,
    observer: str = "minmax",
    disable_bias_correction: bool = False,
    analyze: bool = False,
    save_quant_onnx_path=None,
    save_quant_params_path=None,
    data_preprocess=None,
    quant_format="qdq",
    device=0,
    disable_quant_names: list = None,
    disable_quant_types: list = None,
    operator_config_by_type: Dict[str, QuantOperatorObserverConfig] = None,
    operator_config_by_name: Dict[str, QuantOperatorObserverConfig] = None,
    passes: Union[str, List, BasePass] = "default",
    verification=None,
    **kwargs,
):
    """
    静态量化一个 ONNX 模型 或 PyTorch 的模型，如果是 PyTorch 的模型，会首先被转为 ONNX 模型，然后再开始进行量化

    :param model: ONNX 文件路径或者 PyTorch  `nn.Module`
    :param calibration_dataloader: 校准数据集 Dataloader
    :param observer: 定点算法，可选项：`minmax`, `hist_percentile`, `percentile`, `entropy`, `ema`
    :param disable_bias_correction: 禁用偏置修正，若为 True，不会修改偏置，否则修改部分偏置
    :param analyze: 启用量化分析
    :param save_quant_onnx_path: 保存量化后的 onnx 到本地的路径
    :param save_quant_params_path: 保存量化参数到本地的路径
    :param data_preprocess: 数据预处理方法
    :param quant_format: 量化输出模型格式
    :param device: 运行设备号
    :param disable_quant_names: 禁止量化的算子名字
    :param disable_quant_types: 禁止量化的算子类型
    :param operator_config_by_type: 通过算子类型去设置对算子的量化方法
    :param operator_config_by_name: 通过算子名字去设置对算子的量化方法
    :param passes: 量化前对 Graph 进行处理的 Pass
    :param verification: 量化模型精度校验类对象，用于校验量化后模型精度，必须实现init, get_data, calculate, summary四个成员函数
    :return: Quantized Graph IR
    """

    if verification is not None:
        if not hasattr(verification, "init") or not callable(
            getattr(verification, "init")
        ):
            raise Exception("verification must has 'init' method.")
        if not hasattr(verification, "get_data") or not callable(
            getattr(verification, "get_data")
        ):
            raise Exception("verification must has 'get_data' method.")
        if not hasattr(verification, "calculate") or not callable(
            getattr(verification, "calculate")
        ):
            raise Exception("verification must has 'calculate' method.")
        if not hasattr(verification, "summary") or not callable(
            getattr(verification, "summary")
        ):
            raise Exception("verification must has 'summary' method.")

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    if data_preprocess is None:
        warnings.warn(
            "static_quantize: `data_preprocess` is none, the default preprocess is used."
        )

        def _default_preprocess(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if torch.is_tensor(x):
                x = x.to(device)

            return x

        data_preprocess = _default_preprocess

    executor = create_executor(backend="torch")
    executor.set_device(device)

    operator_config = create_operator_config(
        observer, operator_config_by_type, operator_config_by_name
    )

    qconfig = QuantizerConfig(operator_config=operator_config)
    qconfig.bias_correction = not disable_bias_correction
    qconfig.quant_analyzer.enable = analyze

    if isinstance(disable_quant_types, (list, tuple)):
        for op_type in disable_quant_types:
            qconfig.operator_config.disable_quantize_with_op(op_type)

    if isinstance(disable_quant_names, (list, tuple)):
        for name in disable_quant_names:
            qconfig.operator_config.disable_quantize_with_op_name(name)

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

    print(
        f"Save quantized model to `{save_quant_onnx_path}` and `{save_quant_params_path}`."
    )

    if quant_format == "qdq":
        pipline = Pipeline(
            create_source(model, example_inputs=example_inputs),
            ToDevice(device=device),
            _create_passes(passes),
            PostTrainingStaticQuantizer(
                calibration_dataloader,
                executor=executor,
                qconfig=qconfig,
                preprocess=data_preprocess,
            ),
            compute_model_acc(
                verification, executor, quant=True, enable=verification is not None
            ),
            AddQdqPair(),
            create_target(
                saved_path=save_quant_onnx_path,
                example_inputs=example_inputs,
                quant_params_path=save_quant_params_path,
                name=model_name,
            ),
        )
    elif quant_format == "ppq":
        pipline = Pipeline(
            create_source(model, example_inputs=example_inputs),
            ToDevice(device=device),
            _create_passes(passes),
            PostTrainingStaticQuantizer(
                calibration_dataloader,
                executor=executor,
                qconfig=qconfig,
                preprocess=data_preprocess,
            ),
            compute_model_acc(
                verification, executor, quant=True, enable=verification is not None
            ),
            create_target(
                saved_path=save_quant_onnx_path,
                example_inputs=example_inputs,
                quant_params_path=save_quant_params_path,
                name=model_name,
            ),
        )
    else:
        raise ValueError("quant_formant only supports ppq and qdq")

    ret = pipline.run()[0]

    executor.remove_outputs_data()
    executor._current_graph.clear_quant_parameters()
    executor._current_graph.clear_var_value()
    executor._current_graph = None
    del pipline
    del executor
    torch.cuda.empty_cache()

    return ret


@torch.no_grad()
def verify_quantized_model(
    model: [str, torch.nn.Module],
    data_preprocess=None,
    quant_format="qdq",
    quant_params: str = None,
    device=0,
    passes: Union[str, List, BasePass] = "default",
    verification=None,
    **kwargs,
):
    """
    校验量化后的模型精度

    :param model: ONNX 文件路径或者 PyTorch  `nn.Module`
    :param data_preprocess: 数据预处理方法
    :param quant_format: 量化输出模型格式
    :param quant_params: ppq量化参数
    :param device: 运行设备号
    :param passes: 量化前对 Graph 进行处理的 Pass
    :param verification: 量化模型精度校验类对象，用于校验量化后模型精度，必须实现init, get_data, calculate, summary四个成员函数
    :return: Quantized Graph IR
    """
    if verification is None:
        warnings.warn("verify_quantized_model: `verification` must be set.")

    if not hasattr(verification, "init") or not callable(getattr(verification, "init")):
        raise Exception("verification must has 'init' method.")
    if not hasattr(verification, "get_data") or not callable(
        getattr(verification, "get_data")
    ):
        raise Exception("verification must has 'get_data' method.")
    if not hasattr(verification, "calculate") or not callable(
        getattr(verification, "calculate")
    ):
        raise Exception("verification must has 'calculate' method.")
    if not hasattr(verification, "summary") or not callable(
        getattr(verification, "summary")
    ):
        raise Exception("verification must has 'summary' method.")

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    if data_preprocess is None:
        warnings.warn(
            "verify_quantized_modele: `data_preprocess` is none, the default preprocess is used."
        )

        def _default_preprocess(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if torch.is_tensor(x):
                x = x.to(device)

            return x

        data_preprocess = _default_preprocess

    executor = create_executor(backend="torch")
    executor.set_device(device)

    if quant_format == "qdq":
        warnings.warn(
            "verify_quantized_model: 'qdq' directed verification hasn't beed suppoted."
        )
        return

        pipline = Pipeline(
            create_source(model),
            LoadQuantParamtersPPQStype(quant_params),
            ToDevice(device=device),
            _create_passes(passes),
            compute_model_acc(
                verification, executor, quant=True, enable=verification is not None
            ),
        )
    elif quant_format == "ppq":
        pipline = Pipeline(
            create_source(model),
            LoadQuantParamtersPPQStype(quant_params),
            ToDevice(device=device),
            _create_passes(passes),
            compute_model_acc(
                verification, executor, quant=True, enable=verification is not None
            ),
        )
    else:
        raise Exception("Unsupport quant_format!")
    pipline.run()
