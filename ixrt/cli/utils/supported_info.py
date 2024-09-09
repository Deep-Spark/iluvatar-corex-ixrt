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

from tabulate import tabulate
from ixrt import DataType, get_supported_op_info

__all__ = ["process_supported_ops_info"]

_dtype_map = {
    DataType.INT8: "INT8",
    DataType.HALF: "FP16",
    DataType.FLOAT: "FP32",
    DataType.FLOAT64: "FP64",
    DataType.UINT8: "UI8",
    DataType.INT32: "I32",
    DataType.INT64: "I64",
    DataType.BOOL: "BOOL",
}


def precisions_to_str(precisions) -> str:
    result = [_dtype_map[p] for p in precisions]
    return ",".join(sorted(result, reverse=True))


def make_supported_info(include_plugin=True):
    import ixrt.utils.load_ixrt_plugin

    info = get_supported_op_info()
    onnx_ops = sorted(list(info.onnx_ops))
    plugin_ops = sorted(list(info.plugin_ops))
    precision_map = info.precision_map

    all_data = []
    for op in onnx_ops:
        precision_str = precisions_to_str(precision_map[op])
        if not precision_str:
            continue
        all_data.append(dict(Ops=op, Precision=precision_str, Note=""))

    if include_plugin:
        for op in plugin_ops:
            all_data.append(dict(Ops=op, Precision="", Note="IxRT Plugin"))
    return all_data


def to_table(tablefmt="pretty", path=""):
    onnx_ops_table = tabulate(
        make_supported_info(),
        tablefmt=tablefmt,
        headers="keys",
        showindex="always",
        numalign="left",
        stralign="left",
    )

    if path:
        with open(path, "w") as f:
            f.write(onnx_ops_table)
    else:
        print(onnx_ops_table)


def process_supported_ops_info(path_or_choice):
    if path_or_choice.endswith(".csv"):
        to_table("tsv", path_or_choice)
        print("Save to", path_or_choice)
    elif path_or_choice:
        to_table(path_or_choice)
    if path_or_choice:
        exit(0)
