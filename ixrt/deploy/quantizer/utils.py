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

import numpy as np
from ixrt.deploy.core.progress_bar import progress_bar
from ixrt.deploy.ir import BaseExecutor

try:
    import torch
except:
    torch = None


def tensor_to_list(tensor, to_scaler: bool = False):
    if torch is not None and torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()

    if isinstance(tensor, np.ndarray):
        tensor = tensor.tolist()
        if to_scaler and isinstance(tensor, (tuple, list)) and len(tensor) == 1:
            return tensor[0]
        return tensor

    return tensor


def to_numpy(tensor):
    if torch is not None and torch.is_tensor(tensor):
        return tensor.cpu().numpy()

    if not isinstance(tensor, np.ndarray):
        return np.array(tensor)

    return tensor


def run_graph_one_epoch(
    executor: BaseExecutor, graph, dataloader, preprocess=None, desc: str = None
):
    preprocess = preprocess if preprocess is not None else lambda x: x
    dataloader = progress_bar(dataloader, desc=desc)

    for inputs in dataloader:
        if preprocess is not None:
            inputs = preprocess(inputs)
        executor.execute_graph(graph, inputs)
