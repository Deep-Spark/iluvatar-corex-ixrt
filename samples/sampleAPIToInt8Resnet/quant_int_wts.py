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
import copy
import json
import os
import random
import struct
import time
from argparse import ArgumentParser
from os.path import basename, dirname, join
from random import shuffle
from typing import Any, Dict, Iterable, Tuple, Type

import cv2
import numpy as np
import onnx
import torch
import torch.fx as fx
import torch.nn as nn
import torchvision.datasets
from calibration_dataset import getdataloader
from onnx import numpy_helper
from ixrt.deploy import static_quantize

data_path = os.path.join(dirname(__file__), "../../../data/resnet18_int8")


def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.
        if (
            node.op != "call_module"
        ):  # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if (
            type(modules[node.target]) is nn.BatchNorm2d
            and type(modules[node.args[0].target]) is nn.Conv2d
        ):
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


def benchmark(inp, model, iters=20):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time() - begin)


def export_onnx():
    import torchvision

    net = torchvision.models.resnet18(pretrained=True)
    net.eval()
    fused_net = fuse(net)

    inp = torch.randn(1, 3, 224, 224)

    onnx_path = os.path.join(data_path, "vision_resnet18.onnx")
    torch.onnx.export(
        fused_net,  # model being run
        inp,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def quant_onnx(dataset_dir):
    setseed()
    model_path = os.path.join(data_path, "vision_resnet18.onnx")
    save_dir = data_path
    step = 200
    bsz = 1
    imgsz = 224
    observer = "hist_percentile"
    calibration_dataloader = getdataloader(dataset_dir, step, bsz, img_sz=imgsz)
    print("end of calibration_dataloader")
    static_quantize(
        model_path,
        calibration_dataloader=calibration_dataloader,
        save_quant_params_path=os.path.join(
            save_dir, f"quantized_vision_resnet18.json"
        ),
        observer=observer,
        data_preprocess=lambda x: x[0].to("cuda"),
        quant_format="ppq"
        # quant_format="qdq"
    )
    print("end of static_quantize")


def save_wts():
    onnx_path = os.path.join(data_path, "vision_resnet18.onnx")
    quant_json_path = os.path.join(data_path, "quantized_vision_resnet18.json")
    save_path = os.path.join(data_path, "resnet18_fusebn_ppq_int8.wts")

    model = onnx.load(onnx_path)
    weights = model.graph.initializer
    tensor_dict = {}
    for w in weights:
        tensor_dict[w.name] = np.frombuffer(w.raw_data, np.float32).reshape(w.dims)

    with open(quant_json_path, "r") as f:
        quant_info = json.load(f)["quant_info"]
    for node in model.graph.node:
        node_name = node.name
        if node.op_type == "Conv":
            weight_name = node.input[1]
            node_name = weight_name[:-7]
        for idx, output_name in enumerate(node.output):
            tensor_dict[node_name + "_output_amax_" + str(idx)] = np.array(
                [quant_info[output_name]["tensor_max"]]
            )

    for idx, input in enumerate(model.graph.input):
        tensor_dict["input_amax_" + str(idx)] = np.array(
            [quant_info[input.name]["tensor_max"]]
        )

    f = open(save_path, "w")

    f.write("{}\n".format(len(tensor_dict)))
    for k, v in tensor_dict.items():
        v = v.reshape(-1)
        f.write("{} {}".format(k, len(v)))
        for vv in v:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    f.close()


def create_argparser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument(
        "--imagenet_val", type=str, default="/home/datasets/cv/ImageNet/val"
    )
    return parser


if __name__ == "__main__":
    export_onnx()
    parser = create_argparser()
    args = parser.parse_args()
    quant_onnx(args.imagenet_val)
    save_wts()
