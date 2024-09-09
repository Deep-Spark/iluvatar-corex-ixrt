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

import torch

from ...ir import Graph, GraphTransform
from ..base_pass import BasePass, registe_pass


@registe_pass(level=2)
class ShowQuantGraphPass(BasePass):
    """将量化参数添加作为算子的输入，可以在将 Onnx 导出时查看量化参数"""

    def __init__(self, prefix: str = None):
        if prefix is None:
            prefix = "scale."
        self.prefix = prefix

    def process(self, graph: Graph) -> Graph:
        if isinstance(graph, (tuple, list)):
            graph = graph[0]
        self.transform = GraphTransform(graph)
        for op in graph.operators.values():
            for var in op.inputs + op.outputs:
                if not graph.is_quant_variable(var):
                    continue
                quant_param = graph.get_quant_parameter(var)
                scale = quant_param.scale
                if isinstance(scale, (tuple, list)):
                    scale = torch.tensor(scale)
                scale = self.transform.make_variable(f"{self.prefix}{var}", value=scale)
                op.inputs.append(scale.name)

        return graph
