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

from collections import OrderedDict, namedtuple
from typing import Mapping, Union

from tabulate import tabulate
from ixrt.deploy.utils.comparison_expression import compare_with_op

_ResultDType = Union[Mapping, int, float]

EvaluatorResult = namedtuple(
    "EvaluatorResult",
    ["is_best", "is_satisfied", "best_params", "best_result", "best_value"],
)


class AutotuningEvaluator(object):
    def __init__(
        self, optimized_target: str = "max", metric_name: str = None, threshold=None
    ):
        self.target = optimized_target
        self.metric_name = metric_name
        self.threshold = threshold

        self.results = []
        self.params = []
        self.best_param = None
        self.best_result = None
        self.best_value = None

        if optimized_target == "max":
            self.cmp_op = "gt"
        elif optimized_target == "min":
            self.cmp_op = "lt"
        else:
            raise RuntimeError(
                f"Invalid target `{optimized_target}`, only support `min` or `max`."
            )

    def reset(self):
        self.results.clear()
        self.params.clear()

    def evaluate(self, param, result):
        self.params.append(param)
        self.results.append(result)

        is_best = self.compare(result, self.best_result)

        if is_best:
            self.best_param = param
            self.best_result = result

        return EvaluatorResult(
            is_best=is_best,
            is_satisfied=False
            if self.threshold is None or self.best_value is None
            else self.compare(self.best_value, self.threshold),
            best_params=self.best_param,
            best_result=self.best_result,
            best_value=self.best_value,
        )

    def compare(self, current: _ResultDType, previous: _ResultDType):
        if isinstance(current, Mapping):
            if self.metric_name is None:
                raise RuntimeError(
                    "The argument `metric_name` must be given when the type of result is dict."
                )

            current = current[self.metric_name]

            if previous is not None:
                previous = previous[self.metric_name]

        if previous is None:
            self.best_value = current
            return True

        is_best = compare_with_op(self.cmp_op, current, previous)

        if is_best:
            self.best_value = current

        return is_best

    def summary(self) -> str:
        if len(self.params) == 0:
            return ""

        results = self.results
        if not isinstance(self.results[0], Mapping):
            results = [{"Matric": v} for v in self.results]

        headers = list(self.params[0].keys()) + list(results[0].keys())
        table_data = OrderedDict()

        def unpack_data(v):
            if isinstance(v, (tuple, list)) and len(v) == 1:
                return v[0]
            return v

        for col_name in headers:
            values = self.params if col_name in self.params[0] else results
            table_data[col_name] = [unpack_data(v[col_name]) for v in values]

        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt="pretty",
            showindex="always",
            numalign="left",
            stralign="left",
        )
        return table_str
