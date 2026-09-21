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

import os


def get_edge_path(root, name):
    return os.path.join(root, name.replace("/", "$") + ".npy")


def output_name_alias(name):
    if name.endswith("_output_0"):
        return name[:-2]
    if name.endswith("_output"):
        return name + "_0"
    return None


def alias_pair_occupied(name, alias, universes):
    if not alias:
        return False
    return any(name in universe and alias in universe for universe in universes)


def collect_onnx_tensor_names(onnx_path):
    names = set()
    if not onnx_path or not os.path.isfile(onnx_path):
        return names
    try:
        import onnx

        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:
        return names
    graph = model.graph
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.name:
            names.add(value.name)
    for node in graph.node:
        names.update(n for n in node.input if n)
        names.update(n for n in node.output if n)
    return names
