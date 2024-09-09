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

from collections import OrderedDict
from pprint import pprint
from typing import List, Optional

from ixrt.deploy.ir import Operator


class PGNode(object):
    def __init__(self, op_type, name=None, parents=None, childs=None, operator=None):
        self.op_type = op_type
        self.name = id(self) if name is None else name
        self.operator = operator

        parents = [parents] if isinstance(parents, PGNode) else parents
        childs = [childs] if isinstance(childs, PGNode) else childs

        self.parents = [] if parents is None else parents
        self.childs = [] if childs is None else childs

    def set_operator(self, operator):
        self.operator = operator

    def get_operator(self) -> Operator:
        return self.operator

    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)

    def add_child(self, node):
        if node not in self.childs:
            self.childs.append(node)

    def __eq__(self, other):
        return other.name == self.name

    def to_dict(self):
        return OrderedDict(
            name=self.name,
            op_type=self.op_type,
            operator=self.operator,
            parents=[node.name for node in self.parents],
            childs=[node.name for node in self.childs],
        )


class PatternGraph(object):
    def __init__(self):
        self._nodes = OrderedDict()

    @property
    def nodes(self) -> List[PGNode]:
        return list(self._nodes.values())

    @property
    def node_names(self) -> List[str]:
        return list(self._nodes.keys())

    @property
    def root(self) -> Optional[PGNode]:
        for node in self._nodes.values():
            if len(node.parents) == 0:
                return node
        return None

    def get_node(self, name) -> PGNode:
        return self._nodes.get(name, None)

    def build_node(self, op_type, name=None, parents=None, childs=None) -> PGNode:
        if name == None:
            name = self.generate_node_name(op_type)
        node = PGNode(op_type, name, parents, childs)

        parents = [parents] if isinstance(parents, PGNode) else parents
        childs = [childs] if isinstance(childs, PGNode) else childs

        parents = [] if parents is None else parents
        childs = [] if childs is None else childs

        for p in parents:
            self.add_edge(p, node)

        for child in childs:
            self.add_edge(node, child)

        self.add_node(node)
        return node

    def add_node(self, node):
        if node.name not in self._nodes:
            self._nodes[node.name] = node

    def add_edge(self, src: PGNode, dst: PGNode):
        """src -> dst"""
        self.add_node(src)
        self.add_node(dst)

        src.add_child(dst)
        dst.add_parent(src)

    def to_dict(self):
        node_dict = [node.to_dict() for node in self.nodes]
        return dict(nodes=node_dict)

    def containe_node(self, name):
        return name in self.nodes

    def generate_node_name(self, op_type: str):
        if op_type not in self.node_names:
            return op_type

        count = 0
        pattern = op_type + "_{i}"
        while 1:
            name = pattern.format(i=count)
            if name not in self.node_names:
                return name
            count += 1


def build_sequence_graph(op_types):
    parent = None
    graph = PatternGraph()
    for op_type in op_types:
        parent = graph.build_node(op_type, parents=parent)
    return graph


if __name__ == "__main__":
    g = PatternGraph()

    conv1 = g.build_node("Conv")
    bn1 = g.build_node("BN", parents=conv1)

    conv2 = g.build_node("Conv")
    bn2 = g.build_node("BN", parents=conv2)

    cat = g.build_node("Cat", parents=[bn1, bn2])
    pprint(g.to_dict())

    g = build_sequence_graph(["Conv", "BN", "Relu"])
    pprint(g.to_dict())
