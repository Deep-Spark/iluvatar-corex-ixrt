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

from typing import List, Optional, Tuple, Union

import torch
from ixrt.deploy.core.hook import PriorityHooks
from ixrt.deploy.core.progress_bar import progress_bar

from ...ir import BaseExecutor, ExecutorHook, Graph, Operator, Variable
from ...ir.data_type_mapping import torch_to_ir_dtype
from ...ir.operator_type import OperatorType as OP
from ...visualize import add_row_seperat_line, tabulate
from ..analyzer import QuantAnalyzer, compute_quantization_error
from ..observer import IDENTITY_OBSERVER, HistObserver, QuantVariableObserver
from ..observer.quant_observer import IdentityObserverType
from ..quant_operator import get_registed_quant_operators
from ..quant_operator_config import QuantOperatorObserverConfig
from ..quant_paramter import QuantParameter
from .bias_correction import BiasCorrection
from .quantizer_config import QuantizerConfig
from .quantizer_hook import QuantizerHook


def is_quant_dtype(var, quant_dtypes):
    value = var.value
    if value is None:
        return False

    if not torch.is_tensor(value):
        return False
    dtype = torch_to_ir_dtype(value.dtype)
    return dtype in quant_dtypes


class HistObserverHook(QuantizerHook):
    def __init__(self):
        super(HistObserverHook, self).__init__()
        self._hist_observers = []

    def on_init_end(self):
        self.detect_hist_observer()

    def on_finetune_start(self):
        for var in self.quantizer.graph.variables.values():
            if isinstance(var.value_observer, HistObserver) and var.value is not None:
                var.value_observer.finished_find_minmax()
                var.value = var.value
                var.value_observer.start_find_minmax()

        if len(self._hist_observers) != 0:
            print("Starting collect minmax ...")

    def on_finetune_end(self):
        if len(self._hist_observers) != 0:
            for observer in self._hist_observers:
                observer.finished_find_minmax()
            print("Starting generate histogram ...")
            self.quantizer.finetune(name="GenerateHistogram")

    def detect_hist_observer(self):
        for var in self.quantizer.graph.variables.values():
            if isinstance(var.value_observer, HistObserver):
                self._hist_observers.append(var.value_observer)


class BiasCorrectionHook(QuantizerHook):
    def on_quantize_end(self):
        if not self.quantizer.qconfig.bias_correction:
            return

        self.bias_correction = BiasCorrection(
            self.quantizer.graph, self.quantizer.executor
        )
        self.bias_correction.start_collect_quant_error()
        with self.quantizer.executor.enable_quant_context():
            self.quantizer.finetune(name="BiasCorrectionFinetune")
        self.bias_correction.correct()


class QuantAnalyzerHook(QuantizerHook):
    def on_quantize_end(self):
        self.quantizer: BaseQuantizer
        analyzer_config = self.quantizer.qconfig.quant_analyzer

        if not analyzer_config.enable:
            return

        analyer = QuantAnalyzer(
            graph=self.quantizer.graph,
            executor=self.quantizer.executor,
            metric=analyzer_config.metric,
            reduce_elements=analyzer_config.reduce_elements,
            reduce_samples=analyzer_config.reduce_samples,
            error_level=analyzer_config.error_level,
        )
        self.quantizer.analyzer = analyer

        analyer.start_collect_quant_error()

        if analyer.error_level.lower() == "layer":
            with self.quantizer.executor.disable_quant_context():
                self.quantizer.finetune(
                    name="QuantAnalyzerFinetune",
                    dataloader=self.quantizer.val_dataloader,
                )
        else:
            with self.quantizer.executor.enable_quant_context():
                self.quantizer.finetune(
                    name="QuantAnalyzerFinetune",
                    dataloader=self.quantizer.val_dataloader,
                )
        analyer.finished()
        analyer.print(topk=analyzer_config.show_topk)


class ClearNonFloatingTensorObserverHook(QuantizerHook):
    def __init__(self, quant_dtypes: list):
        super(ClearNonFloatingTensorObserverHook, self).__init__()
        self._hook = self._executor_hook(quant_dtypes)
        self.quant_dtypes = quant_dtypes

    def on_finetune_start(self):
        self._hook._first_forward = True
        self._hook.removed_observer_vars = []
        self.quantizer.executor.add_hook(hook=self._hook)

    def on_finetune_end(self):
        self.quantizer.executor.remove_hook(hook=self._hook)

        vars = set(self._hook.removed_observer_vars)
        if len(vars) != 0:
            print(f"Clear observer of non-floating tensor, variables: {vars}.")

    def _executor_hook(self, quant_types):
        class _Hook(ExecutorHook):
            def __init__(self):
                super(_Hook, self).__init__()
                self._first_forward = True
                self.removed_observer_vars = []

            def on_exec_operator_start(self, operator):
                if not self._first_forward:
                    return

                for input in operator.inputs:
                    variable = self.graph.get_variable(input)
                    if (
                        not is_quant_dtype(variable, quant_types)
                        and variable.value_observer is not None
                    ):
                        self.removed_observer_vars.append(input)
                        variable.remove_value_observer()

            def on_exec_operator_end(self, operator, outputs):
                if not self._first_forward:
                    return

                for output in operator.outputs:
                    variable = self.graph.get_variable(output)
                    if (
                        not is_quant_dtype(variable, quant_types)
                        and variable.value_observer is not None
                    ):
                        self.removed_observer_vars.append(output)
                        variable.remove_value_observer()

            def on_exec_graph_end(self):
                self._first_forward = False

        return _Hook()


class DeleteDisabledOperatorObserver(QuantizerHook):
    def on_init_end(self):
        for op in self.quantizer.graph.operators.values():

            if not self.quantizer.qconfig.operator_config.is_disable_operator(op):
                continue

            for var in op.inputs:
                var: Variable = self.quantizer.graph.get_variable(var)
                var.remove_value_observer()


class ShareQuantParamsHook(QuantizerHook):

    ONLY_PER_TENSOR_TYPES = [
        OP.EXPAND,
        OP.FLATTEN,
        OP.GATHER,
        OP.GATHER_ELES,
        OP.GATHER_ND,
        OP.PERMUTE,
        OP.SCATTER,
        OP.SCATTER_ND,
        OP.GATHER_ND,
        OP.SCATTER_ELES,
        OP.SLICE,
        OP.SPLIT,
        OP.SPLIT_TO_SEQUENCE,
        OP.SQUEEZE,
        OP.TRANSPOSE,
        OP.UNSQUEEZE,
    ]

    ONLY_FIRST_OUTPUT_TYPES = [
        OP.MAX_POOL,
        OP.REDUCE_MAX,
        OP.REDUCE_MIN,
        OP.TOPK,
    ]

    def __init__(self, shared_types: List):
        super().__init__()
        self.shared_types = shared_types

    def on_quantize_end(self):
        graph = self.quantizer.graph
        for op in graph.toposort():
            if op.op_type in self.shared_types:
                self.share_quant_params(graph, op)
        return graph

    def share_quant_params(self, graph, operator):
        input_quant_params: QuantParameter = graph.get_quant_parameter(
            operator.inputs[0], default=None
        )
        if input_quant_params is None:
            return

        if (
            input_quant_params.per_channel
            and operator.op_type in self.ONLY_PER_TENSOR_TYPES
        ):
            return

        outputs = operator.outputs

        if operator.op_type in self.ONLY_FIRST_OUTPUT_TYPES:
            outputs = outputs[:1]

        for output in outputs:
            if output not in graph.quant_parameters:
                continue

            graph.add_quant_parameter(output, input_quant_params)


class BaseQuantizer(object):
    def __init__(
        self,
        executor: BaseExecutor,
        qconfig: QuantizerConfig = None,
        hooks=None,
        val_dataloader=None,
        show_quant_config=True,
    ):
        self.executor: BaseExecutor = executor
        self.graph: Optional[Graph] = None
        self.qconfig: QuantizerConfig = qconfig or QuantizerConfig.default_config()
        self.val_dataloader = val_dataloader
        self.show_quant_config = show_quant_config

        self.analyzer: Optional[QuantAnalyzer] = None
        self._hist_observers: List[HistObserver] = []

        self.hooks: Union[PriorityHooks, QuantizerHook] = PriorityHooks()
        self.add_hook(ClearNonFloatingTensorObserverHook(self.qconfig.quant_dtype), 10)
        self.add_hook(HistObserverHook(), 1)
        self.add_hook(BiasCorrectionHook(), -9999)
        self.add_hook(QuantAnalyzerHook(), -9999)
        # self.add_hook(DeleteDisabledOperatorObserver(), -9999)
        self.add_hook(ShareQuantParamsHook(qconfig.share_quant_params_types), -9999)

        if isinstance(hooks, QuantizerHook):
            self.add_hook(hooks)
        elif isinstance(hooks, (tuple, list)):
            for hook in hooks:
                self.add_hook(hook)

    def get_matched_quant_operator(self, op: Operator):
        return get_registed_quant_operators().get(op.op_type, None)

    def get_matched_operator_observer_config(
        self, op: Operator
    ) -> Optional[QuantOperatorObserverConfig]:
        if self.qconfig.operator_config.is_disable_operator(op):
            return None

        config = self.qconfig.operator_config.get_config_with_op_name(op.name)
        if config is not None:
            return config

        config = self.qconfig.operator_config.get_config_with_op(op.op_type)
        if config is not None:
            return config

        config = self.qconfig.operator_config.get_global_config()
        if config is not None:
            return config

        return None

    def __call__(self, graph: Graph):
        self.graph = graph
        self._set_hook_state()

        self.hooks.on_quantize_start()

        self.hooks.on_init_start()
        self.init_observer()
        self.hooks.on_init_end()

        if self.show_quant_config:
            self.print_quant_config()

        self.hooks.on_finetune_start()
        self.finetune()
        self.hooks.on_finetune_end()

        self.hooks.on_convert_start()
        self.convert()
        self.hooks.on_convert_end()

        self.clear_observer()

        self.hooks.on_quantize_end()

        return self.graph

    def init_observer(self):
        for op_name, op in self.graph.operators.items():
            op.unmark_as_quant_op()

            quant_outputs = self._is_quant_output(op)

            quant_op = self.get_matched_quant_operator(op)
            if quant_op is None:
                continue

            operator_observer_config = self.get_matched_operator_observer_config(op)
            if operator_observer_config is None:
                continue

            op.mark_as_quant_op()

            quant_op(
                self.graph, op, operator_observer_config, quant_outputs=quant_outputs
            )

        self._hit_init_value_for_quant_var()

    def finetune(self, name=None, dataloader=None):
        pass

    def convert(self):
        progress = progress_bar(self.graph.variables.items(), desc="CollectQuantParam")
        for name, quant_var in progress:
            quant_params = self._get_quant_params(quant_var)
            if quant_params is not None:
                self.graph.add_quant_parameter(name, quant_params)

    def clear_observer(self):
        for var in self.graph.variables.values():
            if self.is_quant_variable(var):
                var.remove_value_observer()

    def is_quant_variable(self, var: Variable):
        return (
            isinstance(var.value_observer, QuantVariableObserver)
            and var.value_observer != IDENTITY_OBSERVER
        )

    def add_hook(self, hook, priority=None):
        self.hooks.add_hook(hook, priority)

    def remove_hook(self, hook):
        self.hooks.remove_hook(hook)

    def _set_hook_state(self):
        for hook in self.hooks.hooks.values():
            hook = hook.hooks
            for _hook in hook:
                _hook.set_quantizer(self)

    def print_quant_config(self):
        headers = ["Layer Name", "Activation", "Weight", "Bias"]
        config_str = []

        format_pattern = "{observer}\n{grain}\n"

        def format_observer(observer: QuantVariableObserver):
            if observer is None:
                return "Identity"
            if isinstance(observer, IdentityObserverType):
                return "Identity"
            return format_pattern.format(
                observer=observer.__class__.__name__,
                grain=""
                if observer.quant_policy is None
                else observer.quant_policy.grain.name,
            )

        for op in self.graph.operators.values():
            qconfig = [op.name, "Identity", "Identity", "Identity"]

            for input in op.inputs:
                input = self.graph.get_variable(input)
                observer = input.value_observer
                if not isinstance(observer, QuantVariableObserver):
                    continue
                if observer.is_activation():
                    qconfig[1] = format_observer(observer)
                elif observer.is_weight():
                    qconfig[2] = format_observer(observer)
                elif observer.is_bias():
                    qconfig[3] = format_observer(observer)
            config_str.append(qconfig)

        config_str = add_row_seperat_line(config_str)
        config_str.pop(-1)
        print(tabulate(config_str, headers=headers, tablefmt="pretty"))

    def _get_quant_params(self, var: Variable):
        if not self.is_quant_variable(var):
            return None
        return var.value_observer.get_quant_parameters()

    def _hit_init_value_for_quant_var(self):
        for var in self.graph.variables.values():
            if self.is_quant_variable(var) and var.value is not None:
                if is_quant_dtype(var, self.qconfig.quant_dtype):
                    var.value = var.value
                else:
                    var.remove_value_observer()

    def _is_quant_output(self, operator):
        if self.qconfig.use_qat:
            return False

        # 判断算子是否是图中的用于输出的算子
        next_ops = self.graph.get_next_operators(operator)
        if len(next_ops) == 0:
            return True

        # 判断算子的下一层是否需要量化，如果不需要量化
        for next_op in next_ops:
            next_op_quant_config = self.get_matched_operator_observer_config(next_op)
            if next_op_quant_config is None or isinstance(
                next_op_quant_config.activation, IdentityObserverType
            ):
                return True

        return False
