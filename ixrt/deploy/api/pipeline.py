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

from ixrt.deploy.ir import BaseSource

from ..backend.torch.executor.torch_executor import to_device


class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, graph):
        return to_device(graph, self.device)


class Pipeline(object):
    def __init__(self, source: BaseSource, *components):
        self.source = source
        self.components = list(components)
        self.origin_graph = self.source()
        self.out_graph = None
        self.to_device()

    def add(self, comp):
        self.components.append(comp)

    def delete(self, comp):
        if comp in self.components:
            self.components.remove(comp)

    def to_device(self):
        for comp in self.components:
            if isinstance(comp, ToDevice):
                self.origin_graph = comp(self.origin_graph)
                break

    def __call__(self):
        return self.run()

    def run(self):
        graph = self.origin_graph
        for comp in self.components:
            graph = comp(graph)
        self.out_graph = graph
        return graph
