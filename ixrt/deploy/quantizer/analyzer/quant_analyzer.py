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

import inspect
from collections import OrderedDict
from typing import Callable, Dict, Optional, Union

import torch
from ixrt.deploy.visualize import tabulate

from ...ir import Graph, Operator
from ...ir.base_executor import BaseExecutor
from ...ir.executor_hook import ExecutorHook, LambdaExecutorHook
from ..utils import run_graph_one_epoch
from .analyzer_metrics import batch_reduce, get_metric


class BaseAnalyzerCollector(object):
    def __init__(self, metric: str, reduce_elements: str, reduce_samples: str):
        self.metric = get_metric(metric) if isinstance(metric, str) else metric
        self.reduce_elements = reduce_elements
        self.reduce_samples = reduce_samples
        self._op_quant_error = OrderedDict()

    def start_collect(self):
        pass

    def finished(self):
        self.get_errors()

    def reset(self):
        self._op_quant_error.clear()

    def update(
        self, operator: Operator, real_out: torch.Tensor, quant_out: torch.Tensor
    ):
        if torch.is_tensor(real_out) and torch.is_tensor(quant_out):
            self._update_op_out_tensor(operator, real_out, quant_out)
        elif isinstance(real_out, (tuple, list)) and isinstance(
            quant_out, (tuple, list)
        ):
            if len(real_out) != len(quant_out):
                print(
                    f"Warning: Skip collect {operator.name}({operator.op_type}) error, "
                    f"because the number of outputs is not same."
                )
                return

            for rt, qt in zip(real_out, quant_out):
                if torch.is_tensor(rt) and torch.is_tensor(qt):
                    self._update_op_out_tensor(operator, rt, qt)

        else:
            print(
                f"Warning: Skip collect {operator.name}({operator.op_type}) error, "
                f"because the output type of operator is not supported, got {type(quant_out)}."
            )

    def _update_op_out_tensor(
        self, operator: Operator, real_out: torch.Tensor, quant_out: torch.Tensor
    ):
        if not torch.is_tensor(real_out) or not torch.is_tensor(quant_out):
            return

        samples_error = self.metric(real_out, quant_out, self.reduce_elements)
        samples_error = samples_error.reshape(1, -1)
        samples_error = (
            batch_reduce(samples_error, self.reduce_samples).reshape(-1).detach().cpu()
        )

        if not operator.name in self._op_quant_error:
            self._op_quant_error[operator.name] = []

        self._op_quant_error[operator.name].append(samples_error)

    def get_errors(self) -> OrderedDict:
        return self._op_quant_error


class LayerwiseCollector(BaseAnalyzerCollector):
    def __init__(self, executor: BaseExecutor, graph: Graph, *args, **kwargs):
        super(LayerwiseCollector, self).__init__(*args, **kwargs)
        self.executor = executor
        self.graph = graph
        self._collected_outs = dict()
        self._exec_op_hook = LambdaExecutorHook(
            LambdaExecutorHook.on_exec_operator_end, self.on_exec_operator_end
        )

    def start_collect(self):
        self.executor.add_hook(self._exec_op_hook)
        self._op_quant_error.clear()

    def finished(self):
        self.executor.remove_hook(self._exec_op_hook)
        return super(LayerwiseCollector, self).finished()

    def on_exec_operator_end(self, operator: Operator, outputs):
        if not operator.is_quant_operator:
            return

        self.executor: BaseExecutor
        if self.executor.enable_qaunt:
            real_out = self.executor.exec_nonquant_operator(self.graph, operator)
            quant_out = outputs
        else:
            real_out = outputs
            quant_out = self.executor.exec_quant_operator(self.graph, operator)

        self.update(operator, real_out, quant_out)
        return outputs


class GraphwiseCollector(
    BaseAnalyzerCollector,
):
    def __init__(self, executor: BaseExecutor, graph: Graph, *args, **kwargs):
        super(GraphwiseCollector, self).__init__(*args, **kwargs)
        self.executor = executor
        self.graph = graph
        self._graph_copy: Optional[Graph] = None
        self._exec_op_hook = ExecutorHook()
        self._exec_op_hook.on_exec_graph_start = self.on_exec_graph_start
        self._exec_op_hook.on_exec_operator_end = self.on_exec_operator_end

    def start_collect(self):
        self.executor.add_hook(self._exec_op_hook)
        self._op_quant_error.clear()
        self._graph_copy = self.graph.copy()

    def finished(self):
        self._graph_copy = None
        self.executor.remove_hook(self._exec_op_hook)
        return super(GraphwiseCollector, self).finished()

    def on_exec_graph_start(self):
        for input in self.graph.inputs.values():
            self._graph_copy.set_var_value(input.name, input.value)

    def on_exec_operator_end(self, operator: Operator, outputs):
        if self.executor.enable_qaunt:
            self.executor._current_graph = self._graph_copy
            real_out = self.executor.exec_nonquant_operator(self._graph_copy, operator)
            self.executor._current_graph = self.graph
            self.executor.set_output(self._graph_copy, operator, real_out)
            quant_out = outputs
        else:
            real_out = outputs
            self.executor._current_graph = self._graph_copy
            quant_out = self.executor.exec_quant_operator(self._graph_copy, operator)
            self.executor._current_graph = self.graph
            self.executor.set_output(self._graph_copy, operator, quant_out)

        self.update(operator, real_out, quant_out)

        return outputs


class QuantAnalyzer(object):
    def __init__(
        self,
        graph: Graph = None,
        executor: BaseExecutor = None,
        metric: Union[str, Callable] = "l2",
        reduce_elements="mean",
        reduce_samples="mean",
        error_level: str = "layer",
    ):
        """
        :param graph: IR Graph
        :param executor: Torch Executor
        :param metric: 计算误差的指标，可选：l1, l2
        :param reduce_elements: 对于一个输入Shape为[Batch, C, H, W]，在 C, H, W 维度上以什么方式去归约
               可选的值有：sum, min, max, mean
        :param reduce_samples: 经过 reduce_elements 之后得到的Shape[Batch]，
               以什么方式去归约得到最后的误差
        :param error_level: 可选的值有：layer 和 graph，
               layer 表示每一层的误差，其中每一层的输出是 FP32；
               graph 表示整体的误差，其中每一层的输出是 量化后的输出，
                      该种模式下表示前一层的量化误差会累计到下一层
        """
        self.graph = graph
        self.executor: BaseExecutor = executor
        self.metric_name = metric
        self.metric = get_metric(metric) if isinstance(metric, str) else metric
        self.reduce_elements = reduce_elements
        self.reduce_samples = reduce_samples
        self.error_level = error_level

        self.op_quant_error = OrderedDict()

        if error_level.lower() == "layer":
            self.collector = LayerwiseCollector(
                executor,
                graph,
                metric=self.metric,
                reduce_elements=self.reduce_elements,
                reduce_samples=self.reduce_samples,
            )
        elif error_level.lower() == "graph":
            self.collector = GraphwiseCollector(
                executor,
                graph,
                metric=self.metric,
                reduce_elements=self.reduce_elements,
                reduce_samples=self.reduce_samples,
            )
        else:
            raise RuntimeError("Invalid error level, expect `layer` or `graph`")

    def set_graph(self, graph):
        self.graph = graph
        self.collector.graph = graph

    def set_executor(self, executor):
        self.executor = executor
        self.collector.executor = executor

    def start_collect_quant_error(self):
        if self.graph is None:
            raise RuntimeError("The graph is none, please call set_graph function.")

        if self.executor is None:
            raise RuntimeError(
                "The executor is none, please call set_executor function."
            )

        self.collector.start_collect()

    def finished(self):
        self.collector.finished()
        self.op_quant_error = self.collector.get_errors()

    def reset(self):
        self.collector.reset()
        self.op_quant_error.clear()

    def report(self) -> Dict[str, float]:
        op_names = self.op_quant_error.keys()
        op_errors = dict()

        for op_name in op_names:
            errors = torch.stack(self.op_quant_error[op_name]).reshape(1, -1)
            errors = batch_reduce(errors, self.reduce_samples)
            op_errors[op_name] = errors.reshape(-1).item()

        return op_errors

    def print(self, topk: int = None):
        quant_error = self.report()
        quant_error = sorted(quant_error.items(), key=lambda x: x[1], reverse=True)
        if topk is not None:
            quant_error = quant_error[:topk]

        metric_name = self.metric_name
        if hasattr(metric_name, "__name__") and not isinstance(metric_name, str):
            metric_name = metric_name.__name__
        if inspect.isclass(metric_name):
            metric_name = metric_name.__class__.__name__

        print(tabulate(quant_error, headers=["Layer Name", f"Error ({metric_name})"]))


def compute_quantization_error(analyer: QuantAnalyzer, dataloader, preprocess=None):
    analyer.start_collect_quant_error()

    if analyer.error_level.lower() == "layer":
        with analyer.executor.disable_quant_context():
            run_graph_one_epoch(
                analyer.executor,
                analyer.graph,
                dataloader,
                preprocess=preprocess,
                desc="ErrorAnalysis",
            )
    else:
        with analyer.executor.enable_quant_context():
            run_graph_one_epoch(
                analyer.executor,
                analyer.graph,
                dataloader,
                preprocess=preprocess,
                desc="ErrorAnalysis",
            )

    analyer.finished()
    return analyer.report()
