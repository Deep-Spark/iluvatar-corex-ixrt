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

from typing import Callable, List, Union

import ixrt.deploy

from ..graph import Graph
from .base_transform import BaseGraphTransform


class GraphSearcherMix(BaseGraphTransform):
    def find_sequence_subgraph(
        self,
        pattern,
        callback,
        strict=True,
    ):
        """
        查找一串序列的子图
        :param pattern: 一组算子类型的列表
        :param callback: 匹配成功后的回调函数
        :param strict: 是否是严格模式，如果是严格模式，那么序列子图中不存在其他的节点，如果不是，那么匹配到的图与模式图会存在枝干
        """
        # type: (Union[List[str], ixrt.deploy.fusion.PatternGraph], Callable[[Graph, ixrt.deploy.fusion.PatternGraph], None], bool) -> None
        from ...fusion.matcher import GraphMatcher, PatternGraph, build_sequence_graph

        if isinstance(pattern, List):
            pattern = build_sequence_graph(pattern)

        matcher = GraphMatcher(pattern, strict=strict)
        return matcher.findall(self.graph, callback)
