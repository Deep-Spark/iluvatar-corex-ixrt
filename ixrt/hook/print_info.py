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

__all__ = ["print_info"]


def print_info(info):
    headers = ["name", "shape", "paddings", "dtype", "format"]
    itensors = info.input_tensors
    otensors = info.output_tensors
    print("Op:", info.op_name, "\ninputs:")
    data = []
    for i in range(info.nb_inputs):
        data.append(
            [
                info.input_names[i],
                tuple(itensors[i].shape),
                tuple(itensors[i].paddings),
                itensors[i].dtype,
                itensors[i].format,
            ]
        )
    try:
        print(tabulate(data, headers=headers, tablefmt="grid"))
    except Exception as e:
        print(e)
    print("outputs:")
    data = []
    for i in range(info.nb_outputs):
        data.append(
            [
                info.output_names[i],
                tuple(otensors[i].shape),
                tuple(otensors[i].paddings),
                otensors[i].dtype,
                itensors[i].format,
            ]
        )
    print(tabulate(data, headers=headers, tablefmt="grid"))
    print("\n\n")
