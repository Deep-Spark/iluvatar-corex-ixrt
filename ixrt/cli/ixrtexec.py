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

#! /usr/local/bin python3

import json
import os
import shutil
import sys
import time
from ctypes import CDLL
from os.path import abspath, dirname, join
from time import sleep

import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np
import onnx
from onnx import helper, TensorProto
import ixrt
from tabulate import tabulate
from ixrt.cli.compare_result import IxrtLayerSaver, create_acc_comp_config
from ixrt.cli.utils import (
    args_parser,
    check_cuda_errors,
    generate_input_buffers,
    generate_output_buffers,
    generate_quantified_model,
    get_cuda_error_enum,
    get_io_info,
    parse_custom_inputs,
    parse_inference_input_shapes,
)

from ixrt.hook import create_hook
import fcntl
log_info_dict = {
    "verbose": ixrt.Logger.VERBOSE,
    "info": ixrt.Logger.INFO,
    "warning": ixrt.Logger.WARNING,
    "error": ixrt.Logger.ERROR,
    "internal_error": ixrt.Logger.INTERNAL_ERROR,
}


def check_status(ret, msg):
    if not ret:
        raise Exception(msg)


def convert_opset_version(onnx_path, save_path, opset=11):
    # load ONNX model
    onnx_model = onnx.load(onnx_path)
    model_opset = onnx_model.opset_import[0].version
    if model_opset != opset:
        # convert model
        try:
            converted_model = onnx.version_converter.convert_version(onnx_model, opset)
            # save converted model
            onnx.save(converted_model, save_path)
            return True
        except:
            return False
    else:
        return False


def change_onnx_input_dim(model, input_shapes):
    inputs = model.graph.input
    for input in inputs:
        if input.name not in input_shapes:
            continue
        input_shape = input_shapes[input.name]
        dim_vals = input.type.tensor_type.shape.dim
        for i, dim in enumerate(dim_vals):
            dim.dim_param = str(input_shape[i])
            dim.dim_value = input_shape[i]


def restore_inputs_dimensions(src_path, dst_path):
    src_dim_vals_map = {}
    src_model = onnx.load(src_path)
    for input in src_model.graph.input:
        dim_vals = input.type.tensor_type.shape.dim
        src_dim_vals_map[input.name] = dim_vals
    dst_model = onnx.load(dst_path)
    for input in dst_model.graph.input:
        dim_vals = input.type.tensor_type.shape.dim
        del dim_vals[:]
        src_dim_vals = src_dim_vals_map[input.name]
        dim_vals.MergeFrom(src_dim_vals)

    onnx.save(dst_model, dst_path)


def create_serialized_network_by_build(exec_config):
    onnx_path = exec_config.onnx
    print(
        "It is highly recommended to use onnxsim(https://pypi.org/project/onnxsim/) to simplify the model before converting the ONNX model with ixrtexec, as this can often enhance inference performance."
    )
    converted_model = ".tmp.converted_optset11.onnx"
    status = convert_opset_version(onnx_path, converted_model)
    if status:
        onnx_path = converted_model

    ixrt_input_shapes, is_dynamic_model, input_type_dict, _ = get_io_info(exec_config)
    if exec_config.precision not in ["fp16", "bf16", "int8", "fp32"]:
        raise ValueError(f"unsupported precision {exec_config.precision}")

    import_quantified_qdq_model = False
    internally_quantified_qdq_model = None

    if exec_config.precision == "int8" and exec_config.quant_file is not None:
        if not os.path.isfile(exec_config.quant_file):
            raise ValueError(f" no such file path {exec_config.quant_file}")

    elif exec_config.precision == "int8":
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        for node in graph.node:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                import_quantified_qdq_model = True
                break
        if not import_quantified_qdq_model:
            calibration_tensor_shape = {}
            dynamic_min_opt_max_shape = (
                {}
                if "dynamic_input" not in ixrt_input_shapes
                else ixrt_input_shapes["dynamic_input"]
            )
            for tensor_name in ixrt_input_shapes:
                if tensor_name == "dynamic_input":
                    continue
                if tensor_name in dynamic_min_opt_max_shape:
                    calibration_tensor_shape[tensor_name] = dynamic_min_opt_max_shape[
                        tensor_name
                    ].min_shape
                else:
                    calibration_tensor_shape[tensor_name] = ixrt_input_shapes[
                        tensor_name
                    ]
            internally_quantified_qdq_model = ".tmp.quant.onnx"
            internally_quantified_qdq_model_params = ".tmp.quant.params.json"
            generate_quantified_model(
                onnx_path,
                calibration_tensor_shape,
                input_type_dict,
                internally_quantified_qdq_model,
                internally_quantified_qdq_model_params,
            )
            restore_inputs_dimensions(onnx_path, internally_quantified_qdq_model)


    IXRT_LOGGER = ixrt.Logger(log_info_dict[exec_config.log_level])
    builder = ixrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(ixrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    timing_cache = create_timing_cache(build_config, exec_config.timingCacheFile)
    if timing_cache:
        build_config.set_timing_cache(timing_cache, False)
    build_config.builder_optimization_level = exec_config.builderOptimizationLevel

    parser = ixrt.OnnxParser(network, IXRT_LOGGER)

    parse_status = True
    if exec_config.precision == "int8":
        if import_quantified_qdq_model:
            parse_status = parser.parse_from_file(onnx_path)
        elif internally_quantified_qdq_model is not None:
            parse_status = parser.parse_from_file(internally_quantified_qdq_model)
        else:
            parse_status = parser.parse_from_files(onnx_path, exec_config.quant_file)
        build_config.set_flag(ixrt.BuilderFlag.INT8)
    elif exec_config.precision == "fp16":
        parse_status = parser.parse_from_file(onnx_path)
        build_config.set_flag(ixrt.BuilderFlag.FP16)
    elif exec_config.precision == "bf16":
        parse_status = parser.parse_from_file(onnx_path)
        build_config.set_flag(ixrt.BuilderFlag.BF16)
    else:
        parse_status = parser.parse_from_file(onnx_path)
    assert parse_status, "Failed to parse model, please go back to check error message!"
    opt_profile = builder.create_optimization_profile()
    if is_dynamic_model:
        for name, desc in ixrt_input_shapes["dynamic_input"].items():
            opt_profile.set_shape(name, desc.min_shape, desc.opt_shape, desc.max_shape)
    else:
        for input_index in range(network.num_inputs):
            inp = network.get_input(input_index)
            input_name = inp.name
            input_shape = ixrt_input_shapes[input_name]
            inp.shape = input_shape
            opt_profile.set_shape(input_name, input_shape, input_shape, input_shape)
    build_config.add_optimization_profile(opt_profile)
    build_config.profiling_verbosity = ixrt.ProfilingVerbosity.DETAILED

    plan = builder.build_serialized_network(network, build_config)
    if os.path.exists(converted_model):
        os.remove(converted_model)
    save_timing_cache(build_config, exec_config.timingCacheFile)
    return plan


def read_profiler_data(profiler_context):
    lines = []
    import io
    buf = io.StringIO(profiler_context)
    lines = buf.readlines()

    loops_num = int(lines[0])
    layers_num = int(lines[1])
    kernels_num = int(lines[2])

    layer_names = lines[3 : 3 + layers_num]
    layer_ids = lines[3 + layers_num : 3 + 2 * layers_num]
    layer_end = 3 + 2 * layers_num + loops_num * layers_num
    layer_times = lines[3 + 2 * layers_num : layer_end]
    kernel_names = lines[layer_end : layer_end + kernels_num]
    kernel_layer_ids = lines[layer_end + kernels_num : layer_end + 2 * kernels_num]
    kernel_times = lines[
        layer_end
        + 2 * kernels_num : layer_end
        + 2 * kernels_num
        + kernels_num * loops_num
    ]

    layer_names = [name.strip() for name in layer_names]
    layer_ids = [int(id_.strip()) for id_ in layer_ids]
    layer_times = [int(time.strip()) for time in layer_times]
    kernel_names = [name.strip() for name in kernel_names]
    kernel_layer_ids = [int(id_.strip()) for id_ in kernel_layer_ids]
    kernel_times = [int(time.strip()) for time in kernel_times]

    return [
        layer_names,
        layer_ids,
        layer_times,
        kernel_names,
        kernel_layer_ids,
        kernel_times,
    ]


def export_profiler_data(exec_config, infos):
    (
        layer_names,
        layer_ids,
        layer_times,
        kernel_names,
        kernel_layer_ids,
        kernel_times,
        layer_infos,
    ) = infos
    layer_len = len(layer_names)
    kernel_len = len(kernel_names)

    layer_id_name = dict()
    layer_datas = np.array(layer_times).reshape((-1, layer_len))
    for i, layer_id in enumerate(layer_ids):
        layer_id_name[layer_id] = layer_names[i]

    layer_avg_time = (
        np.mean(layer_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    layer_sum_time = (
        np.sum(layer_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    layer_medium_time = (
        np.median(layer_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    layer_sum_time_percent = layer_sum_time / np.sum(layer_sum_time)

    # process kernel data
    kernel_name = [""] * kernel_len
    kernel_name_show = [""] * kernel_len
    kernel_layer_name = [""] * kernel_len
    kernel_datas = np.array(kernel_times).reshape((-1, kernel_len))
    for i, layer_id in enumerate(kernel_layer_ids):
        if layer_id < 0:
            continue
        kernel_name[i] = kernel_names[i].replace(",", ";")
        kernel_name_show[i] = (
            kernel_names[i]
            .split("(")[0]
            .replace("void ", "")
            .replace("cuinfer::impl::kernel::", "")
            .replace("nvinfer1::inferrt::", "")
            .replace("iluvatar::inferrt::", "")[0:70]
        )
        kernel_layer_name[i] = layer_id_name[layer_id]

    kernel_avg_time = (
        np.mean(kernel_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    kernel_sum_time = (
        np.sum(kernel_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    kernel_medium_time = (
        np.median(kernel_datas[exec_config.warmUp :, :], axis=0) / 1000.0 / 1000.0
    )
    kernel_sum_time_percent = kernel_sum_time / np.sum(kernel_sum_time)

    # layer info show on screen
    show_table = []
    for i in range(layer_len):
        show_table.append(
            [
                layer_names[i],
                layer_sum_time[i],
                layer_avg_time[i],
                layer_medium_time[i],
                layer_sum_time_percent[i] * 100,
            ]
        )
    print(f"\n=== Profile Layer ({exec_config.iterations} iterations) ===")
    print(
        tabulate(
            show_table,
            floatfmt=".3f",
            colalign=("right",),
            headers=[
                "Layer Name",
                "Time(ms)",
                "Avg. Time(ms)",
                "Median Time(ms)",
                "Time(%)",
            ],
        )
    )
    print(" ")

    # kernel info show on screen
    show_table = []
    for i in range(kernel_len):
        show_table.append(
            [
                kernel_layer_name[i],
                kernel_name_show[i],
                kernel_sum_time[i],
                kernel_avg_time[i],
                kernel_medium_time[i],
                kernel_sum_time_percent[i] * 100,
            ]
        )
    print(f"\n=== Profile Kernel ({exec_config.iterations} iterations) ===")
    if not exec_config.export_profiler:
        print("Use --export_profiler *.csv get more detail kernel name")
    print(
        tabulate(
            show_table,
            floatfmt=".3f",
            colalign=(
                "right",
                "left",
            ),
            headers=[
                "Layer Name",
                "Kernel Name",
                "Time(ms)",
                "Avg. Time(ms)",
                "Median Time(ms)",
                "Time(%)",
            ],
        )
    )
    print(" ")

    # write to csv file
    if exec_config.export_profiler:
        with open(exec_config.export_profiler, "w+") as f:
            f.write(f"Layer Name,Time(ms),Avg. Time(ms),Median Time(ms),Time(%)\n")
            for i in range(layer_len):
                f.write(
                    f"{layer_names[i]},{layer_sum_time[i]:.3f},{layer_avg_time[i]:.3f},{layer_medium_time[i]:.3f},{layer_sum_time_percent[i] * 100:.2f}\n"
                )
            f.write("\n\n")
            f.write(
                f"Layer Name,Kernel Name,Time(ms),Avg. Time(ms),Median Time(ms),Time(%)\n"
            )
            for i in range(kernel_len):
                f.write(
                    f"{kernel_layer_name[i]},{kernel_name[i]},{kernel_sum_time[i]:.3f},{kernel_avg_time[i]:.3f}, \
                                    {kernel_medium_time[i]:.3f},{kernel_sum_time_percent[i] * 100:.2f}\n"
                )
        print(f"Profile file saved : {exec_config.export_profiler} \n")


def create_engine(exec_config):
    if exec_config.load_engine:
        if not os.path.exists(exec_config.load_engine):
            raise FileNotFoundError(exec_config.load_engine)

        with open(exec_config.load_engine, "rb") as f:
            serialized_network = f.read()
    else:
        serialized_network = create_serialized_network_by_build(exec_config)
        if not serialized_network:
            raise Exception(
                "Failed to build IxRT engine, please use -v to activate verbose logger and then check error messages!"
            )
        if exec_config.save_engine is not None:
            with open(exec_config.save_engine, "wb") as f:
                f.write(serialized_network)
            f.close()

    return serialized_network


def load_plugins(logger, exec_config):
    default_plugins = {
        "ixrt_plugin": join(dirname(__file__), "..", "lib", "libixrt_plugin.so")
    }
    for plugin_path in exec_config.plugins:
        if plugin_path in default_plugins:
            plugin_path = default_plugins[plugin_path]
        print("Load plugin:", plugin_path)
        CDLL(plugin_path)
        ixrt.init_libnvinfer_plugins(logger, "")

def create_timing_cache(build_config, file_path):
    if not file_path:
        return None
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            try:
                timing_cache = build_config.create_timing_cache(file.read())
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    else:
        timing_cache = build_config.create_timing_cache(b"")
    return timing_cache

def combine_timing_cache(build_config, file_path):
    if not file_path or not os.path.exists(file_path):
        return
    curr_timing_cache = build_config.get_timing_cache()
    other_timing_cache = create_timing_cache(build_config, file_path)
    curr_timing_cache.combine(other_timing_cache, True)
    return curr_timing_cache
def save_timing_cache(build_config, file_path):
    if not file_path:
        return
    timing_cache = build_config.get_timing_cache()
    combine_timing_cache(build_config, file_path)
    with open(file_path, "wb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(timing_cache.serialize())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def main():
    exec_config = args_parser()
    IXRT_LOGGER = ixrt.Logger(log_info_dict[exec_config.log_level])
    load_plugins(IXRT_LOGGER, exec_config)
    if exec_config.run_profiler:
        os.environ["IXRT_USE_PROFILER"] = "1"
    serialized_network = create_engine(exec_config)
    runtime = ixrt.Runtime(IXRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_network)
    assert engine
    engine_inspector = engine.create_engine_inspector()
    assert engine_inspector

    if exec_config.dump_graph is not None:
        if "IXRT_DEBUG_TOOL" in os.environ.keys() and os.path.exists(os.environ["IXRT_DEBUG_TOOL"]):
            engine_inspector.save_engine_plan(exec_config.dump_graph)
        else :
            print("developer need to set IXRT_DEBUG_TOOL before use dump_graph")

    context = engine.create_execution_context()
    assert context
    # Use interface hook
    for hook in exec_config.hooks:
        context.register_hook(
            hook, create_hook(hook), ixrt.ExecutionHookFlag.POSTRUN
        )
    if exec_config.verify_acc:
        acc_verify_config = create_acc_comp_config(exec_config)
        exec_config.iterations = 1
        exec_config.warmUp = 0
        ixrt_layer_saver = IxrtLayerSaver(acc_verify_config.ixrt, acc_verify_config.tensors_to_watch)
        context.register_hook(
            "save_layer_output",
            ixrt_layer_saver.save_layer_output_hook,
            ixrt.ExecutionHookFlag.POSTRUN,
        )
        context.register_hook(
            "inject_external_input",
            create_hook(
                "inject_external_input", external_input=acc_verify_config.inject_tensors
            ),
            ixrt.ExecutionHookFlag.PRERUN,
        )
    # Setup I/O bindings
    inference_input_shapes = parse_inference_input_shapes(exec_config)
    bsz = None
    input_bindings = []
    output_bindings = []
    dptrs = []

    # set input shapes
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            engine_dims = engine.get_binding_shape(i)
            is_dynamic_shape = False
            for d in list(engine_dims):
                if d < 0:
                    is_dynamic_shape = True
            if is_dynamic_shape:
                profile_shape = engine.get_profile_shape(0, i)
                if name not in inference_input_shapes:
                    print(
                        "Dynamic dimensions required for input: {}, but no shapes were provided. Automatically overriding shape to: {}".format(
                            name, "x".join([str(i) for i in profile_shape[0]])
                        )
                    )
                    shape = profile_shape[0]
                else:
                    shape = inference_input_shapes[name]
                context.set_input_shape(name, shape)
            else:
                if name in inference_input_shapes:
                    print(
                        "Static dimensions for input: {}, no need to provide shapes. The --shape parameter will be ignored.".format(
                            name
                        )
                    )
            shape = context.get_binding_shape(i)
            print(
                "Input inference shape: {}={}".format(
                    name, "x".join([str(i) for i in shape])
                )
            )

    # After all input shapes set, output shape can be acquired
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if 0 in shape:
            print(f"Warning: Binding '{name}' has a shape with zero element: {shape}. Skipping memory allocation.")
            continue
        if engine.binding_is_input(i):
            if bsz is None:
                bsz = shape[0]
            size = np.dtype(ixrt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
        else:
            size = context.get_max_output_size(name)
        err, dptr = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        binding_data = {
            "name": name,
            "dtype": np.dtype(ixrt.nptype(dtype)),
            "shape": list(shape),
            "dptr": dptr,
            "nbytes": size,
        }
        dptrs.append(dptr)
        if engine.binding_is_input(i):
            input_bindings.append(binding_data)
        else:
            output_bindings.append(binding_data)

    input_buffers = generate_input_buffers(input_bindings, parse_custom_inputs(exec_config))
    output_buffers = generate_output_buffers(output_bindings)
    if exec_config.verify_acc:
        from ixrt.cli.compare_result.ort_layer_saver import OrtLayerSaver

        ort_layer_saver = OrtLayerSaver(acc_verify_config, input_buffers)
        ort_layer_saver.save()
    for input_binding in input_bindings:
        binding_name = input_binding["name"]
        size = input_binding["nbytes"]

        assert size == input_buffers[binding_name].nbytes
        (err,) = cudart.cudaMemcpy(
            input_binding["dptr"],
            input_buffers[binding_name],
            size,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess

    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    if exec_config.warmUp > 0:
        for i in range(exec_config.warmUp):
            check_status(
                context.execute_async_v2(dptrs, stream),
                "Caught error during execution, please check logging message for detailed reason!",
            )
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    start_time = time.time()
    for i in range(exec_config.iterations):
        check_status(
            context.execute_async_v2(dptrs, stream),
            "Caught error during execution, please check logging message for detailed reason!",
        )
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    end_time = time.time()

    (err,) = cudart.cudaStreamDestroy(stream)
    assert err == cudart.cudaError_t.cudaSuccess

    for output_binding in output_bindings:
        binding_name = output_binding["name"]
        size = output_binding["nbytes"]
        (err,) = cudart.cudaMemcpy(
            output_buffers[binding_name],
            output_binding["dptr"],
            size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess

    host_end_time = time.time()

    for input_binding in input_bindings:
        dptr = input_binding["dptr"]
        (err,) = cudart.cudaFree(dptr)
        assert err == cudart.cudaError_t.cudaSuccess
    for output_binding in output_bindings:
        dptr = output_binding["dptr"]
        (err,) = cudart.cudaFree(dptr)
        assert err == cudart.cudaError_t.cudaSuccess

    if exec_config.verify_acc:
        from ixrt.cli.compare_result.compare import compare_ixrt_ort_layer_output

        ok = compare_ixrt_ort_layer_output(
            ixrt_layer_saver,
            ort_layer_saver,
            acc_verify_config,
            [i["name"] for i in output_bindings],
        )
        if not exec_config.save_verify_data:
            shutil.rmtree(acc_verify_config.root)
        exit(0 if ok else 1)
    fps = exec_config.iterations * bsz / (end_time - start_time)
    throughput = exec_config.iterations / (end_time - start_time)
    if not exec_config.run_profiler:
        print(f"fps: {fps} with batchsize {bsz}")
        print(f"Throughput: {throughput} qps")

    if exec_config.run_profiler:
        if exec_config.export_profiler:
            if not exec_config.export_profiler.endswith(".csv"):
                print("[Info] Use command end with .csv : --export_profiler result.csv")
                exit(0)
        profiler_context = context.get_running_profiler()
        if profiler_context != "":
            sleep(0.5)
            if "IXRT_USE_PROFILER" in os.environ.keys():
                del os.environ["IXRT_USE_PROFILER"]
            infos = read_profiler_data(profiler_context)
            layer_names = infos[0]
            layer_infos = {}

            infos.append(layer_infos)

            export_profiler_data(exec_config, infos)
        else:
            print("[Error] Profiler run error !!")
        if "IXRT_USE_PROFILER" in os.environ.keys():
            del os.environ["IXRT_USE_PROFILER"]


if __name__ == "__main__":
    dev_id = 0
    check_cuda_errors(cudart.cudaSetDevice(dev_id))
    main()
