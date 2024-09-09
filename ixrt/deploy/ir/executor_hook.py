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

from typing import Any, Callable, Optional, Union

from ..core.hook import BaseHook


class ExecutorHook(BaseHook):
    def __init__(self):
        super(ExecutorHook, self).__init__()
        self.graph = None
        self.executor = None

    def set_graph(self, graph):
        self.graph = graph

    def set_executor(self, executor):
        self.executor = executor

    def on_exec_graph_start(self):
        pass

    def on_exec_graph_end(self):
        pass

    def on_exec_operator_start(self, operator):
        pass

    def on_exec_operator_end(self, operator, outputs):
        pass


class LambdaExecutorHook(ExecutorHook):
    def __init__(self, name: Union[str, Callable], fn: Callable):
        super(LambdaExecutorHook, self).__init__()
        if not isinstance(name, str):
            if hasattr(name, "__name__"):
                name = name.__name__
            else:
                raise RuntimeError(f"Invalid name `{name}`.")

        if not hasattr(self, name):
            raise RuntimeError(f"Invalid name `{name}`.")

        setattr(self, name, fn)
