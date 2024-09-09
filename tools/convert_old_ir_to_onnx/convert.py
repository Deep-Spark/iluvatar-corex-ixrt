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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Convert IxRT previours intermediate representation to ONNX
## Usage

```bash
python3 convert.py \
            --json_path path/to/graph/json \
            --weight_path path/to/weight \
            --onnx output/onnx/name \
            --input_info input_name1:dtype1:1x2x3,input_name2:dtype2:12x13 --output_info output_name1:dtype1
```

Parameter explained:
- `--json_path`: path to your graph json file
- `--weight_path`: path to your weight file
- `--onnx`: path to save your output onnx file

There may be no "input"/"output" in your previous graph json file, so please use these two arguments to specify model input/output
- `--input_info`: specify model i/o info, format should be `name1:dtype1:shape1,name2:dtype2:shape2`, use `,` to seperate different inputs, use `:` to seperate different attributes. For inputs, shape is a must
- `--output_info`: specify model i/o info, format should be `name1:dtype1:shape1,name2:dtype2:shape2`, use `,` to seperate different outputs, use `:` to seperate different attributes. For outputs, shape is not a must
"""

import argparse
import json
import sys

import onnx
from onnx import TensorProto, helper


def load_json(path):
    with open(path, "r") as f:
        data = f.read()
    return json.loads(data)


def dtype_str_to_onnx(dtype_str):
    mapping = {
        "float16": TensorProto.FLOAT16,
        "float": TensorProto.FLOAT,
        "float32": TensorProto.FLOAT,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
        "bool": TensorProto.BOOL,
    }
    if dtype_str not in mapping:
        raise Exception(
            "mapping not found from str to onnx, tried to convert:", dtype_str
        )

    return mapping[dtype_str]


def get_input_tensor(graph_json, input_info):
    # user defined key has higher priority
    result = []
    if input_info:
        # parse from use defined info
        for i, input_i in enumerate(input_info.split(",")):
            assert (
                len(input_i.split(":")) == 3
            ), f"{i}th input has the format of name:dtype:NxHxWxC, but got {input_i}"
            name, dtype, shape = input_i.split(":")
            shape = [int(i) for i in shape.split("x")]
            value_info = helper.make_tensor_value_info(
                name, dtype_str_to_onnx(dtype), shape
            )
            result.append(value_info)

    elif "input" not in graph_json:
        raise Exception(
            "key (input) is not in graph_json, seems this json file was created from an old version, please specify i/o by yourself!"
        )

    else:
        input_info = graph_json["input"]
        raise NotImplementedError(
            "Parse i/o info from json not implemented yet, this script aims to recover old json+weights"
        )
    return result


def get_output_tensor(graph_json, output_info):
    # user defined key has higher priority
    result = []
    if output_info:
        # parse from use defined info
        for i, output_i in enumerate(output_info.split(",")):
            assert len(output_i.split(":")) in [
                2,
                3,
            ], f"{i}th input has the format of name:dtype:NxHxWxC or just name:dtype, but got {output_i}"
            shape = []
            if len(output_i.split(":")) == 2:
                name, dtype = output_i.split(":")
            else:
                name, dtype, shape = output_i.split(":")
                shape = [int(i) for i in shape.split("x")]
            value_info = helper.make_tensor_value_info(
                name, dtype_str_to_onnx(dtype), shape
            )
            result.append(value_info)

    elif "output" not in graph_json:
        raise Exception(
            "key (input) is not in graph_json, seems this json file was created from an old version, please specify i/o by yourself!"
        )

    else:
        # parse from json
        output_info = graph_json["input"]
        raise NotImplementedError(
            "Parse i/o info from json not implemented yet, this script aims to recover old json+weights"
        )
    return result


def _get_attributes(node_name, node_info):
    if "attrbiute" in node_info:
        return node_info["attrbiute"]
    elif "attribute" in node_info:
        return node_info["attribute"]
    else:
        raise KeyError("No attribute info in node info, node:", node_name)


def get_nodes(graph_json):
    assert (
        "nodes" in graph_json
    ), "key (nodes) is not in graph_json, this json file maybe a wrong file"
    result = []
    nodes = graph_json["nodes"]
    for node_name, node_info in nodes.items():
        op_type = node_info["op_type"]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]
        attrs = _get_attributes(node_name, node_info)
        the_node = helper.make_node(op_type, inputs, outputs, **attrs)
        result.append(the_node)
    return result


def get_dtype_of_edge(tensor_info, edge_info, edge_name):
    if edge_name in tensor_info:
        return tensor_info[edge_name]["data_type"]


def get_shape_of_edge(tensor_info, edge_info, edge_name):
    if edge_name in tensor_info:
        return tensor_info[edge_name]["dims"]


def get_initializer(graph_json, binary_weight):
    assert (
        "tensors" in graph_json
    ), "key (tensors) is not in grpah_json, this json file maybe a wrong file"
    tensor_info = graph_json["tensors"]
    edge_info = graph_json["edges"] if "edges" in graph_json else None

    result = []
    while binary_weight:
        b = binary_weight.read1(8)
        len_name = 0
        if b:
            len_name = int.from_bytes(b, sys.byteorder)
        if not b or not len_name:
            break
        name = binary_weight.read(len_name).decode("utf8").replace("'", '"')
        len_array = int.from_bytes(binary_weight.read(8), sys.byteorder)
        dtype = get_dtype_of_edge(tensor_info, edge_info, name)
        shape = get_shape_of_edge(tensor_info, edge_info, name)
        arr = binary_weight.read(len_array)
        initializer = helper.make_tensor(
            name, dtype_str_to_onnx(dtype), shape, arr, True
        )
        result.append(initializer)
    return result


def get_value_info(graph_json):
    if "edges" not in graph_json:
        return []
    result = []
    edge_info = graph_json["edges"]
    for edge_name, info in edge_info.items():
        shape = info["dims"]
        dtype = dtype_str_to_onnx(info["data_type"])
        value_info = helper.make_tensor_value_info(edge_name, dtype, shape)
        result.append(value_info)
    return result


def main(json_path, weight_path, onnx_path, input_info, output_info):
    graph_json = load_json(json_path)
    print("[INFO] Parse i/o info")
    input_tensor = get_input_tensor(graph_json, input_info)
    output_tensor = get_output_tensor(graph_json, output_info)

    print("[INFO] Parse graph nodes")
    nodes = get_nodes(graph_json)

    print("[INFO] Parse initializers")
    initializers = get_initializer(graph_json, open(weight_path, "rb"))

    print("[INFO] Parse value info")
    value_info = get_value_info(graph_json)

    print("[INFO] graph")
    graph_def = helper.make_graph(
        nodes,
        "Onnx created from IxRT graph json and weights",
        input_tensor,
        output_tensor,
        initializers,
        value_info=value_info,
    )
    model_def = helper.make_model(
        graph_def, producer_name="IxRT", opset_imports=[helper.make_opsetid("", 11)]
    )
    onnx.save(model_def, onnx_path)
    print("[INFO] onnx model to", onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse graph json and weights to onnx")
    parser.add_argument("--json_path", required=True, type=str, help="Graph json path")
    parser.add_argument("--weight_path", required=True, type=str, help="Weight path")
    parser.add_argument(
        "--onnx_path", type=str, help="Onnx path", default="output.onnx"
    )
    parser.add_argument(
        "--input_info",
        type=str,
        help="Input info of the graph, format should be name1:dtype1,name2,dtype2",
        default="",
    )
    parser.add_argument(
        "--output_info",
        type=str,
        help="Output info of the graph, format should be name1:dtype1,name2,dtype2",
        default="",
    )
    config = parser.parse_args()

    main(**config.__dict__)
