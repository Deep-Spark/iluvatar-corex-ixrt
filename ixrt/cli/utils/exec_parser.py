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

import argparse
import os

import onnx


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default=None, required=False, help="ONNX model file path"
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Set input shapes for inference inputs. \
                        Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128 \
                        Each input shape is supplied as a key-value pair where key is the input name and \
                        value is the dimensions (including the batch dimension) to be used for that input. \
                        Each key-value pair has the key and value separated using a colon (:). \
                        Multiple input shapes can be provided via comma-separated key-value pairs.",
    )
    parser.add_argument(
        "--min_shape",
        type=str,
        default="",
        help="Specify the range of the input shapes to build the engine with.",
    )
    parser.add_argument(
        "--opt_shape",
        type=str,
        default="",
        help="Specify the range of the input shapes to build the engine with.",
    )
    parser.add_argument(
        "--max_shape",
        type=str,
        default="",
        help="Specify the range of the input shapes to build the engine with.",
    )
    parser.add_argument(
        "--input_types",
        type=str,
        default=None,
        help="Set input types for inference inputs. \
                        Example input shapes spec: input0:float16,input1:int32",
    )
    parser.add_argument(
        "--output_types",
        type=str,
        default=None,
        help="Set output types for inference outputs. \
                        Example output shapes spec: output0:float32, output1:int32 ",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Run at least N inference iterations"
    )
    parser.add_argument(
        "--warmUp",
        type=int,
        default=10,
        help="Run for N milliseconds to warmup before measuring performance",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["int8", "fp16", "bf16", "fp32"],
        default="fp32",
        help="Set model inference precision, can be set to fp16 or int8",
    )
    parser.add_argument(
        "--quant_file",
        type=str,
        default=None,
        help="quant json file of model, can only used in int8 precision",
    )
    parser.add_argument(
        "--save_engine",
        type=str,
        default=None,
        help="save engine file",
    )
    parser.add_argument(
        "--load_engine",
        type=str,
        default=None,
        help="load engine file",
    )
    parser.add_argument(
        "--dump_graph",
        type=str,
        default=None,
        help="dump engine graph in onnx format",
    )
    parser.add_argument(
        "--run_profiler",
        action="store_true",
        help="running profiling",
    )
    parser.add_argument(
        "--export_profiler",
        type=str,
        default=None,
        help="export profiling data into CSV file(*.csv), need use with '--run_profiler' ",
    )
    parser.add_argument(
        "--support",
        type=str,
        default="",
        choices=["pipe", "pretty"],
        help="Display IxRT supported op info, pass path to csv to save or doing nothing to print",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["verbose", "info", "warning", "error", "internal_error"],
        default="warning",
        help="log level in ixrt",
    )
    parser.add_argument(
        "--load_inputs",
        type=str,
        default="",
        help="Load inputs from file, defaults to generate inputs randomly. Note that input files expected to be binary and format is input_name1:input1.dat,input_name2,input2.dat",
    )
    parser.add_argument(
        "--verify_acc",
        action="store_true",
        help="Verify IxRT's computing acc with 3rd framework (now only onnxruntime)",
    )
    parser.add_argument(
        "--cosine_sim",
        type=float,
        default=0.999,
        help="A float value from 0-1 representing criterion in accuracy comparison, the bigger, the higher accuracy required",
    )
    parser.add_argument(
        "--diff_max",
        type=float,
        default=-1,
        help="Max difference to compare in a tensor. A positive float value used in accuracy comparison, the bigger, the lower accuracy required, if not specified, only use cosine similarity",
    )
    parser.add_argument(
        "--diff_sum",
        type=float,
        default=-1,
        help="Total difference to compare for a tensor. A positive float value used in accuracy comparison, the bigger, the lower accuracy required, if not specified, only use cosine similarity",
    )
    parser.add_argument(
        "--diff_rel_avg",
        type=float,
        default=0.35,
        help="Average relative difference to compare for a tensor. A positive float value used in accuracy comparison, the bigger, the lower accuracy required",
    )
    parser.add_argument(
        "--ort_onnx",
        type=str,
        default="",
        help="The onnx that onnxruntime used to run in accuracy verification. Used when model format to IxRT is difference to that of onnxruntime, for example, engine format or QDQ format to IxRT, thus use this option to pass the original onnx format to onnxruntime",
    )
    parser.add_argument(
        "--ort_cpu",
        action="store_true",
        help="Use CPU instead of CUDA to run onnxruntime, which is only used in accuracy verification.",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        nargs="+",
        metavar="N",
        default=[],
        help="Path to plugin dynamic library to load",
    )
    parser.add_argument(
        "--hooks",
        type=str,
        nargs="+",
        metavar="N",
        default=[],
        help="Hooks to use in runtime execution, a list is allowed to pass, available choices: print_info",
    )
    parser.add_argument(
        "--inject_tensors",
        type=str,
        default="",
        help="Used in accuracy verification. Inject external input tensor to IxRT input. "
        "The format can be either --inject_tensors edge1,edge2 or --inject_tensors edge1:/path/to/edge1.npy,edge2:/path/to/edge2.npy."
        "If specify edge without path, will use result from onnxruntime.",
    )
    parser.add_argument(
        "--save_verify_data",
        type=str,
        default="",
        help="Used in accuracy verification. Save verify middle data in a directory instead delete after run.",
    )
    parser.add_argument(
        "--only_verify_outputs",
        action="store_true",
        help="Used in accuracy verification. When specified, only verify model outputs instead of all layers",
    )
    parser.add_argument(
        "--timingCacheFile",
        type=str,
        default=None,
        help="File path to save timing cache, the algorithm tuning result, to save tuning time for future building engine",
    )
    parser.add_argument(
        "--builderOptimizationLevel",
        type=int,
        default=3,
        help="Set the builder optimization level. (default is 3).\nHigher level allows IxRT to spend more building time for more optimization options.\n"
             "Valid values include integers from 0 to the maximum optimization level, which is currently 5.",
     )
    parser.add_argument(
        "--watch",
        type=str,
        nargs="+",
        help="Used in accuracy verification. When specified, watch the specified edge from producing until consuming, in order to verify memory crash",
    )
    config = parser.parse_args()
    return config


onnx_data_type_dict = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
    17: "FP8",
}


def get_tensor_info(tensor):
    tensor_info = {}
    tensor_info["name"] = tensor.name
    tensor_type = tensor.type.tensor_type
    tensor_elem_type = tensor_type.elem_type
    tensor_info["type"] = onnx_data_type_dict[tensor_elem_type]
    tensor_shape = []
    for dim in tensor_type.shape.dim:
        index = dim.WhichOneof("value")
        if index:
            shape_val = getattr(dim, index)
        else:
            shape_val = -1
        tensor_shape.append(shape_val)
    tensor_info["shape"] = tensor_shape
    return tensor_info


def onnx_parser(onnx_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    model_inputs = {}
    initializer_list = [init.name for init in graph.initializer]
    for tensor in graph.input:
        if tensor.name not in initializer_list:
            model_inputs[tensor.name] = get_tensor_info(tensor)
    model_outputs = {}
    for tensor in graph.output:
        model_outputs[tensor.name] = get_tensor_info(tensor)

    return model_inputs, model_outputs


if __name__ == "__main__":
    args = args_parser()
    onnx_parser(args.onnx)
