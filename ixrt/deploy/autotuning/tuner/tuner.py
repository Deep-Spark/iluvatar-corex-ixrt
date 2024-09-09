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

import time
import traceback
from typing import Callable, List, Union

from ..paramter import ParameterSpace
from ..tuner_logger import get_tuner_logger
from .tuning_evaluator import AutotuningEvaluator, EvaluatorResult

tuner_logger = get_tuner_logger()


class Tuner(object):
    def __init__(self, evaluator: AutotuningEvaluator = None, ignore_exception=False):
        if evaluator is None:
            evaluator = AutotuningEvaluator()
        self.evaluator = evaluator
        self.ignore_exception = ignore_exception

        self.best_result: EvaluatorResult = None

    def tune(
        self,
        target_fn: Callable,
        space: ParameterSpace,
        callback: Callable[[int, dict, EvaluatorResult], None] = None,
        **kwargs,
    ) -> EvaluatorResult:
        """
        对目标函数在 ParameterSpace 中进行搜索，每次搜索的结果可以通过 callback 来获取
        :param target_fn: 函数的参数必须与参数空间中的定义完全匹配
        :param space: 参数空间
        :param callback: callback(step, params, evaluator_result), 回调函数
        :param kwargs: target_fn 的额外参数，会被一起传入到 target_fn 的参数中
        :return: 返回搜索过程中最佳的一组参数
        """

        space_size = len(space)
        tune_start_time = time.time()
        step_start_time = tune_start_time
        for idx, params in enumerate(space):
            step_end_time = time.time()
            eta = int(step_end_time - step_start_time) * (space_size - idx)
            step_start_time = step_end_time

            print("\n")
            tuner_logger.info(f"Task: {idx}/{space_size}  ETA: {eta}s")

            try:
                result = target_fn(**params, **kwargs)
            except Exception as ex:
                if not self.ignore_exception:
                    raise ex

                print(traceback.print_exc())
                result = ex.__class__.__name__

            evaluate_result = self.evaluator.evaluate(params, result)
            if evaluate_result.is_best:
                self.best_result = evaluate_result

            result_table = self.evaluator.summary()
            tuner_logger.info(f"Running Results:\n{result_table}")  # TODO: Set title

            if callback is not None:
                callback(idx, params, evaluate_result)

            if evaluate_result.is_satisfied:
                tuner_logger.info(
                    "Early stopping, the target of autotuning is satisfied."
                )
                return self.best_result

        tuner_logger.info(f"Best config: {str(self.evaluator.best_param)}")
        tuner_logger.info(f"Best result: {str(self.evaluator.best_result)}")

        return self.best_result
