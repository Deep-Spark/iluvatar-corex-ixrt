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
from abc import ABC, abstractmethod
from collections import defaultdict

from ixrt.deploy.core import Registry

from ..ir import Graph

PASSES_REGISTRY = defaultdict(lambda *args: Registry("PassesRegistry"))


def registe_pass(level, pass_op=None, name=None):
    if pass_op is not None:
        if name is None:
            raise RuntimeError("The argument `name` is required.")
        PASSES_REGISTRY[level].add_handler(name, pass_op)
        return pass_op

    def wrap(pass_cls):
        if not inspect.isclass(pass_cls):
            raise RuntimeError(
                f"The argument `pass_cls` should be a class, got {pass_cls}."
            )

        pass_name = name
        if pass_name is None:
            pass_name = pass_cls.__name__

        PASSES_REGISTRY[level].add_handler(pass_name, pass_cls)
        return pass_cls

    return wrap


def get_pass_registry(level=None):
    if level is None:
        return PASSES_REGISTRY
    return PASSES_REGISTRY[level]


class BasePass(ABC):
    def __call__(self, graph: Graph):
        return self.process(graph)

    @abstractmethod
    def process(self, graph: Graph) -> Graph:
        pass


class PassSequence(BasePass):
    def __init__(self, *passes):
        self.passes = passes

    def process(self, graph: Graph) -> Graph:
        out = graph
        for fusion in self.passes:
            out = fusion(out)

        return out
