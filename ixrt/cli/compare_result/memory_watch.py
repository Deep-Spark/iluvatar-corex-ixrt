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
from __future__ import annotations
import numpy as np
from ixrt.hook.utils import copy_ixrt_tensor_as_np
from dataclasses import dataclass
from .compare import compare_tensor
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

@dataclass
class DiffDetails:
    diff_max: float
    diff_rel_avg: float
    diff_sum: float
    cosine_sim: float
    diff_hist: str
    rel_diff_hist: str
@dataclass
class WatchResult:
    tensor_name: str
    producer: str
    changed_by: str
    consumers: set[str]
    details: DiffDetails
class MemoryWatcher:
    def __init__(self, name: str):
        self._producer = ""
        self._consumers = set()
        self._unvisited_consumers = set()
        self._name = name
        self._tensor_info = None # tensor info description
        self._tensor_data = None # tensor data when being produced
        self._mem_changed = False
        self._mem_changed_by = ""
        self._diff_details = DiffDetails(0., 0., 0., 0., "", "")

    def set_consumers(self, consumers):
        self._unvisited_consumers = consumers

    @property
    def name(self):
        return self._name

    @property
    def mem_changed(self):
        return self._mem_changed

    def mark_as_changed(self, changed_by, details):
        self._mem_changed = True
        self._mem_changed_by = changed_by
        self._diff_details = details

    def watch(self, curr_op_name):
        # not produced yet
        if not self._tensor_info:
            return
        # goal achieved, no need to watch more
        if self._mem_changed:
            return

        # if all consumers meet, stop watching
        if len(self._unvisited_consumers) == 0:
            return

        assert self._tensor_data is not None, "tensor data should be copied once when getting tensor info"
        the_tensor_now = copy_ixrt_tensor_as_np(self._tensor_info, False)
        self._mem_changed = not np.allclose(self._tensor_data, the_tensor_now)

        if self._mem_changed:
            details = DiffDetails(*compare_tensor(self._tensor_data, the_tensor_now))
            self.mark_as_changed(curr_op_name, details)

    def stop_at_consumer(self, consumer):
        if consumer in self._unvisited_consumers:
            self._unvisited_consumers.remove(consumer)

    def record(self, producer, tensor_info):
        self._producer = producer
        self._tensor_info = copy_attrs_to_object(tensor_info)
        self._consumers = tensor_info.consumer_names
        self._tensor_data = copy_ixrt_tensor_as_np(tensor_info, False)
        self._unvisited_consumers = tensor_info.consumer_names

    def get_result(self):
        return WatchResult(self._name, self._producer, self._mem_changed_by, self._consumers, self._diff_details)


class MemoryDiagnosisTool:
    def __init__(self, tensors_to_watch: list[str]):
        self.watchers_ : dict[str, MemoryWatcher]  = dict()
        if tensors_to_watch:
            for name in tensors_to_watch:
                self.watchers_[name] = MemoryWatcher(name)

    def record_since_producing(self, tensor_name, tensor_info, producer):
        if tensor_name not in self.watchers_:
            return

        self.watchers_[tensor_name].record(producer, tensor_info)

    def watch(self, curr_op_name):
        for name, watcher in self.watchers_.items():
            watcher.watch(curr_op_name)
    @property
    def watchers(self) -> dict:
        return self.watchers_
    def stop_at_consumer(self, consumer):
        for name, watcher in self.watchers_.items():
            watcher.stop_at_consumer(consumer)
