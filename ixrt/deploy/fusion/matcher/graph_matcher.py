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

from collections import Counter
from typing import Any, Callable, List

from ixrt.deploy.ir import Graph, Operator

from .pattern_graph import PatternGraph, PGNode


class GraphMatcher(object):
    def __init__(self, pattern_graph: PatternGraph, strict=True):
        self.strict = strict
        self._check_pattern_graph(pattern_graph)
        self.pattern_graph: PatternGraph = pattern_graph

    def findall(self, graph: Graph, callback: Callable[[Graph, PatternGraph], None]):
        root = self.pattern_graph.root
        if root is None:
            raise RuntimeError(
                f"The pattern graph is empty, got {self.pattern_graph.to_dict()}."
            )

        operator_names = list(graph.operators.keys())
        for operator in operator_names:
            if not graph.containe_operator(operator):
                continue
            operator = graph.get_operator(operator)

            if len(self.pattern_graph.nodes) == 1 and root.op_type == operator.op_type:
                root.set_operator(operator)
                callback(graph, self.pattern_graph)
            else:
                self._match_sub_graph(graph, operator, root, callback)

    def _match_sub_graph(
        self,
        graph: Graph,
        ir_op_root: Operator,
        pattern_graph_root: PGNode,
        callback: Callable[[Graph, PatternGraph], None],
    ):
        if ir_op_root.op_type != pattern_graph_root.op_type:
            return False

        pattern_graph_root.set_operator(ir_op_root)

        ir_op_childs = graph.get_next_operators(ir_op_root)
        pnode_childs = pattern_graph_root.childs

        if len(pnode_childs) == 0:
            return True

        if len(ir_op_childs) > 1 and self.strict:
            return False

        for next_ir_op in ir_op_childs:
            sub_graph_status = self._match_sub_graph(
                graph, next_ir_op, pnode_childs[0], callback
            )

            if sub_graph_status:
                callback(graph, self.pattern_graph)

    def _is_same_ops(self, ir_op_childs: List[Operator], pnode_childs: List[PGNode]):
        if len(pnode_childs) != len(ir_op_childs):
            return False

        pnode_childs_count = Counter([node.op_type for node in pnode_childs])
        ir_op_childs_count = Counter([op.op_type for op in ir_op_childs])

        for op_type, count in pnode_childs_count:
            if ir_op_childs_count[op_type] != count:
                return False

        return True

    def _check_pattern_graph(self, pattern_graph: PatternGraph):
        for pnode in pattern_graph.nodes:
            if len(pnode.childs) > 1 or len(pnode.parents) > 1:
                raise RuntimeError("Not support multi-inputs and multi-outputs.")
