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

from ixrt.deploy.ir import Graph

from ..base_pass import BasePass, registe_pass


@registe_pass(level=2)
class ClearUnusedVariablesPass(BasePass):
    """清除 Graph 中未被使用的 变量 和 量化参数"""

    def process(self, graph: Graph) -> Graph:
        vars = list(graph.variables)

        for var in vars:
            if len(graph.get_dst_operators(var)) == 0 and graph.is_leaf_variable(var):
                graph.delete_variable(var)

        quant_params = list(graph.quant_parameters.keys())
        for var in quant_params:
            if not graph.containe_var(var):
                graph.quant_parameters.pop(var)

        return graph
