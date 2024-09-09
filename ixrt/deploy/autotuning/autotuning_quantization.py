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

import abc
from typing import Callable, Dict, List, Optional, Union

from torch.utils.data import DataLoader

from ..autotuning.paramter import *
from ..autotuning.tuner import AutotuningEvaluator, Tuner
from ..autotuning.tuner.tuning_evaluator import EvaluatorResult
from ..backend.torch.executor import TorchExecutor
from ..fusion import BasePass, PassSequence
from ..ir import BaseExecutor, BaseTarget, Graph
from ..quantizer import QuantizerConfig, QuantOperatorObserverConfig
from ..quantizer.algorithm.static_quantizer import PostTrainingStaticQuantizer
from ..quantizer.analyzer import QuantAnalyzer, compute_quantization_error
from ..quantizer.observer import QuantVariableObserver
from ..quantizer.quant_operator_config import create_operator_config

ObserversType = Union[str, Union[List[str], Dict, List[QuantVariableObserver]]]


def create_basic_observers():
    return ["minmax", "hist_percentile", "percentile", "entropy"]


def create_autotuning_observers():
    return [
        dict(name="minmax"),
        dict(name="hist_percentile", percentile=99.999),
        dict(name="hist_percentile", percentile=99.99),
        dict(name="hist_percentile", percentile=99.9),
        dict(name="hist_percentile", percentile=99.0),
        dict(name="percentile", percentile=99.999),
        dict(name="percentile", percentile=99.99),
        dict(name="percentile", percentile=99.9),
        dict(name="percentile", percentile=99.0),
        dict(name="entropy", start_bin=1024 + 512, num_bins=2048),
        dict(name="entropy", start_bin=1024, num_bins=2048),
        dict(name="entropy", start_bin=512, num_bins=2048),
        dict(name="entropy", start_bin=256, num_bins=2048),
        dict(name="entropy", start_bin=2048 + 1024, num_bins=4096),
        dict(name="entropy", start_bin=2048, num_bins=4096),
        dict(name="entropy", start_bin=1024, num_bins=4096),
    ]


def create_basic_param_space(
    activation_observers: ObserversType = None, weight_observers: ObserversType = None
):
    if activation_observers is None:
        activation_observers = create_autotuning_observers()

    if weight_observers is None:
        weight_observers = create_basic_observers()

    if not isinstance(activation_observers, (tuple, list)):
        activation_observers = [activation_observers]

    if not isinstance(weight_observers, (tuple, list)):
        weight_observers = [weight_observers]

    return GridParameterSpace(
        param_list=[
            # 使用 Bias Correction 会修改 Bias 的值，所以如果在搜索过程中加入该选项
            # 会导致后面的搜索是在 Bias Correction 的基础上进行的，可能会产生潜在的问题
            # DiscreteParameter("bias_correction", [False, True]),
            DiscreteParameter("weight_observer", weight_observers),
            DiscreteParameter("activation_observer", activation_observers),
        ]
    )


class AutoMixPrecisionParamSpace(ParameterSpace):
    """
    用于自动混合精度时生成禁止量化的算子，通过每次返回误差最大的算子
    """

    def __init__(self):
        super().__init__([])

        self.graph: Graph = None
        self.disable_quant_ops = []

    def set_graph(self, graph: Graph):
        self.graph = graph

    def next(self):
        if self.graph is None:
            raise RuntimeError(
                "The graph is not given, please call set_graph function."
            )

        if self.result is None:
            return {"disable_quant_ops": self.disable_quant_ops}

        self.disable_quant_ops.append(self.get_next_disable_op())

        return dict(
            disable_quant_ops=self.disable_quant_ops,
        )

    def get_next_disable_op(self) -> Optional[str]:
        self.result: Dict[str, float]
        cand_ops = list(self.result.keys())

        while len(cand_ops) > 0:
            max_error_op = None
            max_error = -float("inf")
            for op in cand_ops:
                if self.result[op] > max_error:
                    max_error_op = op
                    max_error = self.result[op]

            if max_error_op in self.disable_quant_ops:
                cand_ops.remove(max_error_op)
            else:
                return max_error_op

        raise StopIteration()

    def count(self):
        if self.graph is None:
            return 0

        return len(self.graph.operators) + 1


class AutotuningQuantizerEvaluator(AutotuningEvaluator):
    """
    在 Autotuning 过程中依赖于 AutotuningEvaluator 去评估超参数的好坏
    可以继承该类对量化后模型质量的评估
    """

    @abc.abstractmethod
    def evaluate_quantized_model(
        self, executor: BaseExecutor, model: Graph
    ) -> Union[float, dict]:
        """评估模型量化后的质量"""
        pass


class BasedQuantizationAnalyzerEvaluator(AutotuningQuantizerEvaluator):
    """
    使用量化后的模型输出结果 和 FP32 输出结果 的误差去评价模型量化的质量
    """

    def __init__(
        self,
        dataloader: DataLoader,
        preprocess: Callable = None,
        optimized_target: str = "min",
        threshold: float = None,
        analyzer: QuantAnalyzer = None,
        show_results: bool = False,
    ):
        """
        :param dataloader: 用于计算量化误差的数据集
        :param preprocess: Dataloader 的预处理函数
        :param optimized_target: 优化的目标，对于误差而言是最小化
        :param threshold: 阈值，当达到该阈值之后会停止搜索
        :param analyzer: 量化的分析器
        :param show_results: 是否要打印分析的结果
        """
        super().__init__(optimized_target=optimized_target, threshold=threshold)
        self.dataloader = dataloader
        self.preprocess = preprocess
        self.show_results = show_results

        self.analyzer = (
            analyzer
            if analyzer is not None
            else QuantAnalyzer(None, None, metric="l1", error_level="graph")
        )

    def evaluate_quantized_model(self, executor: BaseExecutor, model: Graph):
        analyzer = self.analyzer
        analyzer.set_graph(model)
        analyzer.set_executor(executor)
        analyzer.reset()

        analyzer.set_graph(model)
        quant_error = compute_quantization_error(
            analyzer, self.dataloader, self.preprocess
        )

        if self.show_results:
            analyzer.print()

        output_error = 0
        for output in model.output_names:
            output_op = model.get_src_operator(output).name
            if output_op in quant_error:
                output_error += quant_error[output_op]

        return output_error


class QuantizationErrorEvaluator(BasedQuantizationAnalyzerEvaluator):
    pass


class QuantizationSimilarityEvaluator(BasedQuantizationAnalyzerEvaluator):
    """
    使用量化后的模型输出结果 和 FP32 输出结果 的相似度去评价模型量化的质量
    """

    def __init__(
        self,
        dataloader: DataLoader,
        preprocess: Callable = None,
        threshold: float = 0.999,
        show_results: bool = False,
    ):
        super().__init__(
            dataloader=dataloader,
            preprocess=preprocess,
            optimized_target="max",
            threshold=threshold,
            analyzer=QuantAnalyzer(
                None, None, metric="similarity", error_level="graph"
            ),
            show_results=show_results,
        )


class AutotuningQuantizer(BasePass):
    def __init__(
        self,
        calibration_dataloader: DataLoader,
        executor: BaseExecutor = None,
        evaluator: AutotuningQuantizerEvaluator = None,
        validation_dataloader: DataLoader = None,
        data_preprocess: Callable = None,
        export_target: BaseTarget = None,
        param_space: ParameterSpace = None,
        observers: Union[str, Union[List[str], List[QuantVariableObserver]]] = None,
        disable_quant_names: list = None,
        disable_quant_types: list = None,
        operator_config_by_type: Dict[str, QuantOperatorObserverConfig] = None,
        operator_config_by_name: Dict[str, QuantOperatorObserverConfig] = None,
        automix_precision: bool = False,
        verbose: int = 1,
    ):
        """
        - 自动选择定点算法：会根据 param_space 定义的参数空间去选择定点算法已经量化配置
        - 自动混合精度：会根据每层的量化误差去自动回退算子到 Float 类型进行计算

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
        :param verbose: 用于控制中间日志的状态，值为 0 只显示关键状态的日志，为 1 显示量化的配置，为 2 显示算子的误差
        """

        self.executor = executor if executor is not None else TorchExecutor
        self.calibration_dataloader = calibration_dataloader
        self.validation_dataloader = (
            validation_dataloader
            if validation_dataloader is not None
            else calibration_dataloader
        )
        self.data_preprocess = data_preprocess
        self.export_target = export_target
        self.param_space = (
            param_space
            if param_space is not None
            else create_basic_param_space(observers)
        )
        self.observers = observers
        self.disable_quant_names = disable_quant_names
        self.disable_quant_types = disable_quant_types
        self.operator_config_by_type = operator_config_by_type
        self.operator_config_by_name = operator_config_by_name
        self.automix_precision = automix_precision
        self.verbose = verbose

        self.automix_precision_param_space = AutoMixPrecisionParamSpace()

        self.graph = None
        self.quantized_graph = None

        if isinstance(evaluator, BasedQuantizationAnalyzerEvaluator):
            evaluator.show_error = verbose > 1

        if (
            automix_precision
            and hasattr(evaluator, "threshold")
            and evaluator.threshold is None
        ):
            raise ValueError(
                "The value of threshold must be given when enabling automix precision for autotuning evaluator."
            )

        self.layerwise_analyzer = QuantAnalyzer(
            graph=self.graph, executor=self.executor, metric="l1", error_level="layer"
        )

        if evaluator is None:
            self.evaluator = QuantizationSimilarityEvaluator(
                dataloader=self.validation_dataloader,
                preprocess=data_preprocess,
                show_results=verbose > 1,
            )
        else:
            self.evaluator = evaluator

        self.tuner = Tuner(self.evaluator)

    def reset(self):
        self.evaluator.reset()
        self.layerwise_analyzer.reset()

    def get_best_params(self) -> Optional[EvaluatorResult]:
        if self.tuner.best_result is None:
            return None
        return self.tuner.best_result

    def process(self, graph: Graph) -> Graph:
        return self.quantize(graph)

    def quantize(self, graph: Graph):
        from ixrt.deploy.api.pipeline import ToDevice

        graph = ToDevice(self.executor.default_device())(graph)

        self.graph = graph

        self.automix_precision_param_space.set_graph(graph)
        self.layerwise_analyzer.set_graph(self.graph)

        self.autotuning()
        return graph

    def autotuning(self):
        # 首先将量化中可以搜索的参数进行 Autotuning
        best_result = self.tuner.tune(
            self.quantize_model_with_evaluatation,
            self.param_space,
            callback=self.tuner_callback,
        )

        if best_result is not None and best_result.is_satisfied:
            return

        if best_result is None or best_result.best_params is None:
            best_params = dict()
        else:
            best_params = best_result.best_params

        if not self.automix_precision:
            return

        self.reset()

        # 在得到最佳的量化配置参数之后，根据量化的误差自动回退算子到高精度类型
        self.tuner.tune(
            self.autotuning_op_precision,
            self.automix_precision_param_space,
            callback=self.tuner_callback,
            **best_params,
        )

    def autotuning_op_precision(self, **kwargs):
        result = self.quantize_model_with_evaluatation(**kwargs)

        self.reset()

        quant_error = compute_quantization_error(
            self.layerwise_analyzer, self.validation_dataloader, self.data_preprocess
        )
        self.automix_precision_param_space.set_result(quant_error)

        if self.verbose > 1:
            self.layerwise_analyzer.print()

        return result

    def quantize_model_with_evaluatation(
        self,
        activation_observer=None,
        weight_observer=None,
        bias_correction=False,
        disable_quant_ops=None,
        **kwargs,
    ):
        """
        根据 输入的参数 去量化模型，并返回评估的结果
        返回的结果可以提供一个评价函数去评估，如果没有提供，那么就最小化输出的误差
        """
        self.graph.clear_quant_parameters()

        operator_config = create_operator_config(
            activation_observer,
            self.operator_config_by_type,
            self.operator_config_by_name,
            weight_observer=weight_observer,
        )

        qconfig = QuantizerConfig(operator_config=operator_config)
        qconfig.bias_correction = bias_correction

        if isinstance(self.disable_quant_types, (list, tuple)):
            for op_type in self.disable_quant_types:
                qconfig.operator_config.disable_quantize_with_op(op_type)

        if disable_quant_ops is None:
            disable_quant_ops = []

        if isinstance(self.disable_quant_names, (list, tuple)):
            disable_quant_ops.extend(self.disable_quant_names)

        for name in disable_quant_ops:
            qconfig.operator_config.disable_quantize_with_op_name(name)

        pipeline = PassSequence(
            PostTrainingStaticQuantizer(
                self.calibration_dataloader,
                executor=self.executor,
                qconfig=qconfig,
                preprocess=self.data_preprocess,
                show_quant_config=self.verbose > 0,
            ),
        )
        self.quantized_graph = pipeline.process(self.graph)

        return self.evaluator.evaluate_quantized_model(
            self.executor, self.quantized_graph
        )

    def tuner_callback(self, step: int, params: dict, evaluate_result: EvaluatorResult):
        """
        在 quantize_model_with_evaluatation 运行结束之后，tuner_callback 会被调用
        :param step: 搜索过程中的的步数
        :param params: quantize_model_with_evaluatation 的参数
        :param evaluate_result: 对 quantize_model_with_evaluatation 量化完成之后的评估结果
        """

        if self.export_target is None:
            return

        if evaluate_result.is_best:
            print(f"Save best model on step {step}, params: {params}.")
            self.export_target(self.quantized_graph)
