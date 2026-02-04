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

import functools
from enum import Enum

import numpy as np
from tabulate import tabulate

from .formatting import colorize, green, num2str, red

try:
    from scipy import spatial
except Exception as e:
    raise Exception("Please install scipy if you want to use ixrtexec --verify!")

__all__ = ["compare_ixrt_ort_layer_output"]


class AbnormState(Enum):
    NORMAL_TENSOR = 1
    ZERO_NORM_TENSOR = -1
    NAN_TENSOR = -2
    INF_TENSOR = -3


def get_array_abnormal_state(arr):
    if np.linalg.norm(arr, ord=2) == 0:
        return AbnormState.ZERO_NORM_TENSOR
    elif np.any(np.isnan(arr)):
        return AbnormState.NAN_TENSOR
    elif np.any(np.isinf(arr)):
        return AbnormState.INF_TENSOR
    return AbnormState.NORMAL_TENSOR


def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


def cast_up(buffer):
    dtype = np.dtype(buffer.dtype)

    if dtype == np.dtype(np.float16):
        buffer = buffer.astype(np.float32)
    elif dtype in list(map(np.dtype, [np.int8, np.uint8, np.int16, np.uint16])):
        buffer = buffer.astype(np.int32)
    elif dtype == np.dtype(np.uint32):
        buffer = buffer.astype(np.int64)
    return buffer


def is_empty_shape(shape):
    return volume(shape) == 0


def use_higher_precision(func):
    """
    Decorator that will cast the input numpy buffer(s) to a higher precision before computation.
    """

    @functools.wraps(func)
    def wrapped(*buffers):
        if any(is_empty_shape(buffer.shape) for buffer in buffers):
            return 0

        new_buffers = [cast_up(buffer) for buffer in buffers]
        return func(*new_buffers)

    return wrapped


def compute_max(buffer):
    return np.amax(buffer)


def get_hist_str(output, hist_range=None, title="Histogram"):
    if np.issubdtype(output.dtype, np.bool_):
        return ""

    try:
        try:
            hist, bin_edges = np.histogram(output, range=hist_range)
        except ValueError as err:
            return ""

        max_num_elems = compute_max(hist)
        if not max_num_elems:  # Empty tensor
            return

        bin_edges = [f"{bin:.3g}" for bin in bin_edges]
        max_start_bin_width = max(len(bin) for bin in bin_edges)
        max_end_bin_width = max(len(bin) for bin in bin_edges[1:])

        MAX_WIDTH = 20
        ret = f"---- {title} ----\n"
        ret += f"{'Bin Range':{max_start_bin_width + max_end_bin_width + 5}}|  Num Elems | Visualization\n"
        for num, bin_start, bin_end in zip(hist, bin_edges, bin_edges[1:]):
            bar = "#" * int(MAX_WIDTH * float(num) / float(max_num_elems))
            ret += f"({bin_start:<{max_start_bin_width}}, {bin_end:<{max_end_bin_width}}) | {num:10} | {bar}\n"
        return ret
    except Exception as err:
        raise

def standardize_tensor(a, b):
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    a = a * 1.
    b = b * 1.
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    # if a number both calculated by ixrt or ort is inf or nan, change it to 0
    mask_a = np.isinf(a) | np.isnan(a)
    mask_b = np.isinf(b) | np.isnan(b)
    combined_mask = mask_a & mask_b
    a[combined_mask] = 0
    b[combined_mask] = 0
    return a, b
def compare_tensor(res, ref):
    res, ref = standardize_tensor(res, ref)

    absolute_diff = np.abs(res - ref)
    relative_diff = np.abs(res - ref) / (np.maximum(np.abs(res), np.abs(ref)) + 1e-3)
    diff_max = np.max(absolute_diff.astype(np.float64))
    diff_rel_avg = relative_diff.mean()
    diff_sum = np.sum(absolute_diff.astype(np.float64))

    res_state = get_array_abnormal_state(res)
    ref_state = get_array_abnormal_state(ref)
    if (
        res_state == AbnormState.NORMAL_TENSOR
        and ref_state == AbnormState.NORMAL_TENSOR
    ):
        cosine_sim = 1 - spatial.distance.cosine(res, ref)
    elif (
        res_state == AbnormState.ZERO_NORM_TENSOR
        and ref_state == AbnormState.ZERO_NORM_TENSOR
    ):
        cosine_sim = 1
    elif res_state != AbnormState.NORMAL_TENSOR:
        cosine_sim = red(f"ixrt:{str(get_array_abnormal_state(res))}")
    else:
        cosine_sim = red(f"ort:{str(get_array_abnormal_state(ref))}")

    diff_hist_str = get_hist_str(absolute_diff, title="Absolute diff")
    rel_diff_hist_str = get_hist_str(relative_diff, title="Relative diff")
    return (
        diff_max,
        diff_rel_avg,
        diff_sum,
        cosine_sim,
        diff_hist_str,
        rel_diff_hist_str,
    )


# Relate the edge which is an input or output of reformat nodes of IxRT
# to the actual edge of original graph
def find_possible_common_output(edge_name, ixrt_saver, ort_saver):
    ir = ixrt_saver.runtime_ir
    edge_result_map = ixrt_saver.inference_result

    # go up to find
    ie_to_find = edge_name
    while ie_to_find not in ort_saver.inference_result:
        if ie_to_find not in edge_result_map:
            break
        producer = edge_result_map[ie_to_find].producer_node
        if ir.node_table[producer].type not in ["Cast", "Transpose"]:
            break
        ie_to_find = ir.node_table[producer].input[0]
    else:
        return ie_to_find

    # could not find, go down to find
    oe_to_find = edge_name
    while oe_to_find not in ort_saver.inference_result:
        if oe_to_find not in edge_result_map:
            return None
        consumers = ir.edge_table[oe_to_find].consumers
        if len(consumers) != 1:
            return None
        reformat_consumer = consumers[0]
        if ir.node_table[reformat_consumer].type not in ["Cast", "Transpose"]:
            return None
        oe_to_find = ir.node_table[reformat_consumer].output[0]
    else:
        return oe_to_find

def compare_ixrt_ort_layer_output(ixrt_saver, ort_saver, config, model_outputs):
    ort_result = ort_saver.inference_result
    final_result = True
    error_recorder = ixrt_saver.error_recorder
    print("Start to compare layer output between IxRT and Ort")

    if config.only_verify_outputs:
        ixrt_result = {i: ixrt_saver.inference_result[i] for i in model_outputs}
    else:
        ixrt_result = ixrt_saver.inference_result

    form_data = []
    for edge_name, result in ixrt_result.items():
        ort_edge_name = edge_name
        ixrtpath = result.saved_path
        ixrt_res = np.load(ixrtpath)
        ortpath = None
        ort_res = None
        if ort_edge_name not in ort_result:
            ort_edge_name = find_possible_common_output(
                edge_name, ixrt_saver, ort_saver
            )
            if ort_edge_name:
                ortpath = ort_result[ort_edge_name].saved_path
                ort_res = np.load(ortpath)
                if ixrt_res.dtype != ort_res.dtype:
                    int_compatible_types = [np.dtype("int32"), np.dtype("int64")]
                    float_compatible_types = [np.dtype("float16"), np.dtype("float32"), np.dtype("float64")]
                    if (
                        ixrt_res.dtype in int_compatible_types
                        and ort_res.dtype in int_compatible_types
                    ):
                        ixrt_res = ixrt_res.astype(np.int64)
                        ort_res = ort_res.astype(np.int64)
                    elif (
                        ixrt_res.dtype in float_compatible_types
                        and ort_res.dtype in float_compatible_types
                    ):
                        ixrt_res = ixrt_res.astype(np.float64)
                        ort_res = ort_res.astype(np.float64)
                    else:
                        ort_edge_name = ""

            if not ort_edge_name:
                if config.only_verify_outputs:
                    msg = (
                        f"{edge_name} is output of ixrt, but not exists in onnxruntime"
                    )
                else:
                    msg = f"{edge_name} is not in onnxruntime's graph, skip it"
                error_recorder.append(msg)
                continue
        else:
            ortpath = ort_result[ort_edge_name].saved_path
            ort_res = np.load(ortpath)
        if ixrt_res.shape != ort_res.shape:

            if len(ixrt_res.shape) != len(ort_res.shape):
                print(
                    "incompatible shape between IxRT and Ort for edge:",
                    edge_name,
                    "ixrt:",
                    ixrt_res.shape,
                    "ort:",
                    ort_res.shape,
                )
                continue

            else:
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(ixrt_res.shape, ort_res.shape))
                slices = tuple(slice(0, s) for s in min_shape)
                ort_res = ort_res[slices]
                ixrt_res = ixrt_res[slices]

                print(
                    "Pleace check incompatible shape between IxRT and Ort for edge:",
                    edge_name,
                    "ixrt:",
                    ixrt_res.shape,
                    "ort:",
                    ort_res.shape,
                )

        if ixrt_res.size == 0 and ort_res.size == 0:
            print(
                "empty data in bath IxRT and Ort for edge:",
                edge_name,
                "ixrt:",
                ixrt_res.shape,
                "ort:",
                ort_res.shape,
            )
            continue
        if ort_res.dtype == np.dtype("float64"):
            ort_res = ort_res.astype(np.float32)
        if ixrt_res.dtype != ort_res.dtype:
            int_compatible_types = [np.dtype("int32"), np.dtype("int64")]
            float_compatible_types = [np.dtype("float16"), np.dtype("float32"), np.dtype("float64")]
            if (
                ixrt_res.dtype in int_compatible_types
                and ort_res.dtype in int_compatible_types
            ):
                ixrt_res = ixrt_res.astype(np.int64)
                ort_res = ort_res.astype(np.int64)
            elif (
                ixrt_res.dtype in float_compatible_types
                and ort_res.dtype in float_compatible_types
            ):
                ixrt_res = ixrt_res.astype(np.float64)
                ort_res = ort_res.astype(np.float64)
            else:
                raise Exception(
                    "incompatible dtype between IxRT and Ort for edge:",
                    edge_name,
                    "ixrt:",
                    ixrt_res.dtype,
                    "ort:",
                    ort_res.dtype,
                )

        (
            diff_max,
            diff_rel_avg,
            diff_sum,
            cosine_sim,
            diff_hist_str,
            rel_diff_hist_str,
        ) = compare_tensor(ixrt_res, ort_res)
        if isinstance(cosine_sim, (int, float)):
            is_ixrt_output_right = cosine_sim > config.cosine_sim
        else:
            is_ixrt_output_right = False

        if config.diff_max >= 0:
            is_ixrt_output_right = is_ixrt_output_right and diff_max <= config.diff_max
        if config.diff_sum >= 0:
            is_ixrt_output_right = is_ixrt_output_right and diff_sum <= config.diff_sum
        is_ixrt_output_right = (
            is_ixrt_output_right and diff_rel_avg <= config.diff_rel_avg
        )

        layer_result = green("RIGHT")
        if not is_ixrt_output_right:
            final_result = False
            layer_result = red("WRONG")

        layer_result_str = "{} (from {}) was calculated {} in IxRT, diff_max={}, diff_sum={}, cosine_sim={}"
        diff_sum_str = num2str(diff_sum, (0, config.diff_sum))
        diff_max_str = num2str(diff_max, (0, config.diff_max))
        diff_rel_avg_str = num2str(diff_rel_avg, (0, config.diff_rel_avg))

        if isinstance(cosine_sim, str):
            cosine_sim_str = cosine_sim
        else:
            cosine_sim_str = colorize(cosine_sim, cosine_sim > config.cosine_sim)

        layer_result_str = layer_result_str.format(
            edge_name,
            result.producer_node,
            layer_result,
            diff_max_str,
            diff_sum_str,
            cosine_sim_str,
        )
        # print(layer_result_str)
        measurement_str = "---- Relative Diff Max ----\n{}\n---- Absolute Diff Max ----\n{}\n---- Absolute Diff Sum ----\n{}\n---- Cosine Similarity ----\n{}".format(
            diff_rel_avg_str, diff_max_str, diff_sum_str, cosine_sim_str
        )
        if not is_ixrt_output_right:
            measurement_str += "\n" + "".join([diff_hist_str, rel_diff_hist_str])
        form_data.append(
            [
                layer_result,
                result.producer_node
                + "\n|" * 3
                + "\n"
                + edge_name
                + "\n"
                + str(ixrt_res.shape) + ", " +str(ixrt_res.dtype)
                + "\n",
                measurement_str,
            ]
        )

    print(
        tabulate(
            form_data,
            headers=["Result", "Operator-->Output", "Comparison Results"],
            tablefmt="grid",
            numalign="center",
        )
    )
    result_str = "\033[32mRIGHT\033[0m" if final_result else "\033[31mWRONG\033[0m"
    if error_recorder:
        print(red("Warnings happened during comparison:"))
    for err_msg in error_recorder:
        print(red("- " + err_msg))

    display_memory_crash_check(ixrt_saver.edges_with_memory_crash)
    final_result &= display_memory_watchers(ixrt_saver.mem_diagnosis.watchers)

    print("---------Comparison report---------")
    print("Accuracy compare program has finished!")
    print("Compared with onnxruntime, result calculated from IxRT is", result_str)
    if not final_result:
        print(
            "You may need to begin with checking the first layer which generates the wrong output"
        )
    print("-------------end-------------------")
    return final_result

def display_memory_crash_check(checked_result):
    if not checked_result:
        return

    instruction = "--watch "+" ".join([i.tensor_name for i in checked_result])
    print(red("Memory crashed during comparison, you may need to add parameter to ixrtexec: \n  "+instruction))
    tbl = [[i.tensor_name, i.producer, i.consumer_found_crashed] for i in checked_result]
    print(
        tabulate(
            tbl,
            headers=["Tensor name", "Producer", "Found crashed when consuming"],
            tablefmt="grid",
            numalign="center",
        )
    )
def display_memory_watchers(watchers):
    ret = True
    if watchers:
        print ("Result of tensor memory watcher")
    else:
        return ret

    form_data = []
    for name, watcher in watchers.items():
        watch_result = watcher.get_result()
        if not watch_result.changed_by:
            diagnosis = green("No memory crash found")
        else:
            details = watch_result.details
            measurement_str = f"""
    Before VS After
---- Relative Diff Max ----
{details.diff_rel_avg}
---- Absolute Diff Max ----
{details.diff_max}
---- Absolute Diff Sum ----
{details.diff_sum}
---- Cosine Similarity ----
{details.cosine_sim}
{details.diff_hist}
{details.rel_diff_hist}
"""
            diagnosis = 'changed by -->' + red(watch_result.changed_by) + '<--' + measurement_str
            ret = False

        form_data.append([watch_result.tensor_name, diagnosis, watch_result.producer, "\n".join(watch_result.consumers)])
    print(
        tabulate(
            form_data,
            headers=["Watch", "Diagnosis", "Producer", "Consumers"],
            tablefmt="grid",
            numalign="center",
        )
    )
    return ret