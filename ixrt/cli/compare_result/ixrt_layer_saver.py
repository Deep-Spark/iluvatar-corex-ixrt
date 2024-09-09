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

import dataclasses
import os
import warnings
from collections import OrderedDict, defaultdict
from os.path import exists, join
from typing import AnyStr, List

import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np
import ixrt
from ixrt import LayerType
from ixrt.cli.utils import check_cuda_errors
from ixrt.hook.utils import copy_ixrt_io_tensors_as_np, copy_ixrt_tensor_as_np

from .infer_result import InferResult
from .utils import get_edge_path


def copy_attrs_to_object(obj):
    class GeneratedObject:
        pass

    # 获取对象的所有属性
    obj_attrs = {i: getattr(obj, i) for i in dir(obj) if not i.startswith("_")}

    # 创建属性的拷贝
    obj_attrs_copy = obj_attrs
    print(obj_attrs_copy)

    result = GeneratedObject()
    # 将属性的拷贝转换为对象的属性
    for key, value in obj_attrs_copy.items():
        setattr(result, key, value)
    return result


class Node:
    def __init__(self, type, name, input, output):
        self.type = type
        self.name = name
        self.input = input
        self.output = output


class Edge:
    def __init__(self):
        self.name = None
        self.producer = None
        self.consumers = []


class RuntimeGraph:
    def __init__(self):
        self.node_table = defaultdict(Node)
        self.edge_table = defaultdict(Edge)


class IxrtLayerSaver:
    def __init__(self, ixrt_root="ixrt_layer_output"):
        self.ixrt_root = ixrt_root
        self.inference_result = OrderedDict()
        self.error_recorder = []
        self.runtime_ir = RuntimeGraph()

    def save_layer_output_hook(self, info: ixrt.ExecutionContextInfo):
        check_cuda_errors(cudart.cudaDeviceSynchronize())
        self._collect_info(info)
        self.verify_inputs(info.op_name, info.input_names, info.input_tensors)

        self.save_tensors(
            info.op_name,
            info.input_names,
            info.input_tensors,
            info.nb_inputs,
            info.op_name,
            True,
        )
        self.save_tensors(
            info.op_name,
            info.output_names,
            info.output_tensors,
            info.nb_outputs,
            info.op_name,
            False,
        )

    def save_tensors(self, op_name, names, tensors, n, producer, is_input):
        for i in range(n):
            edge_name = names[i]
            tensor = tensors[i]
            arr = copy_ixrt_tensor_as_np(tensor)
            filename = get_edge_path(self.ixrt_root, edge_name)
            if not is_input:
                np.save(filename, arr)
                self.inference_result[edge_name] = InferResult(
                    producer_node=producer, edge_name=edge_name, saved_path=filename
                )

    def verify_inputs(self, op_name, names, tensors):
        for edge_name, tensor in zip(names, tensors):
            filename = get_edge_path(self.ixrt_root, edge_name)
            arr = copy_ixrt_tensor_as_np(tensor)
            if os.path.exists(filename):
                previous_arr = np.load(filename)
                if not np.allclose(previous_arr, arr):
                    producer_op = self.inference_result[edge_name].producer_node
                    msg = f"{producer_op} --> {edge_name} --> {op_name}, input of lateral operator differs with output of previous operator, memory crash may ocurred"
                    self.error_recorder.append(msg)

    def _collect_info(self, info: ixrt.ExecutionContextInfo):
        self.runtime_ir.node_table[info.op_name] = Node(
            info.op_type, info.op_name, info.input_names, info.output_names
        )
        for oe in info.output_names:
            self.runtime_ir.edge_table[oe].name = oe
            self.runtime_ir.edge_table[oe].producer = info.op_name
        for ie in info.input_names:
            self.runtime_ir.edge_table[ie].name = ie
            self.runtime_ir.edge_table[ie].consumers.append(info.op_name)
