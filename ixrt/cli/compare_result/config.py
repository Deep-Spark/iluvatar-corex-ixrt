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
import tempfile
from os.path import join

from .utils import get_edge_path

__all__ = ["create_acc_comp_config"]


class AccCompConfig:
    root = tempfile.mkdtemp(dir="/tmp")
    ixrt = os.path.join(root, "ixrt")
    ort = os.path.join(root, "ort")
    onnx_path = ""
    cosine_sim: float
    diff_max: float
    diff_rel_avg: float
    diff_sum: float
    ort_cpu: bool
    inject_tensors = {}
    only_verify_outputs: bool


def parse_inject_tensors(exec_config, comp_config):
    if not exec_config.inject_tensors:
        return
    tensors = exec_config.inject_tensors.split(",")
    for t in tensors:
        for i, e in enumerate(t):
            if e == ":":
                if i==len(t):
                    raise Exception(
                        "Wrong use of --inject_tensors, you can use either --inject_tensors edge1,edge2 or --inject_tensors edge1:/path/to/edge1.npy,edge2:/path/to/edge2.npy"
                    )
                elif t[i+1] == ":" or (i>0 and t[i-1]==":"):
                    continue

                file = t[i+1:]
                if not file.endswith(".npy"):
                    comp_config.inject_tensors[t] = get_edge_path(comp_config.ort, file)
                comp_config.inject_tensors[t[:i]] = file
        else:
            comp_config.inject_tensors[t] = get_edge_path(comp_config.ort, t)
    print("Inject tensors:", comp_config.inject_tensors)


def create_acc_comp_config(exec_config):
    result = AccCompConfig()
    if exec_config.save_verify_data:
        result.root = exec_config.save_verify_data
        result.ixrt = join(result.root, "ixrt")
        result.ort = join(result.root, "ort")
    result.onnx_path = (
        exec_config.ort_onnx if exec_config.ort_onnx else exec_config.onnx
    )
    result.cosine_sim = exec_config.cosine_sim
    result.diff_max = exec_config.diff_max
    result.diff_rel_avg = exec_config.diff_rel_avg
    result.diff_sum = exec_config.diff_sum
    result.ort_cpu = exec_config.ort_cpu
    result.inject_tensors = {}
    result.only_verify_outputs = exec_config.only_verify_outputs
    result.tensors_to_watch = exec_config.watch
    os.makedirs(result.ixrt, exist_ok=True)
    os.makedirs(result.ort, exist_ok=True)
    parse_inject_tensors(exec_config, result)
    return result
