/*
 * Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
 * All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License"); you may
 *   not use this file except in compliance with the License. You may obtain
 *   a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *   License for the specific language governing permissions and limitations
 *   under the License.
 */

#pragma once
#include <string>

#include "NvInfer.h"
namespace nvinfer1::samples {
void ResNet18Execute();

void ResNet18Engine();

void ResNet18ExecuteEngine();

void ResNet18DynamicShapeExecute();

void ResNet18ExecuteGPUIO();

void ResNet18ExecuteFP16();

void ResNet18MultiSocket();

void ResNet18MultiThread();

void ResNet18FromOnnx();

void ResNet18OnnxEngine();

void IxRTAPIExecute(const std::string& model_path, const std::string& quant_file = "",
                    const std::string& engine_save_path = "/tmp/resnet18_trt.engine",
                    nvinfer1::BuilderFlag flag = nvinfer1::BuilderFlag::kFP16);

void IxRTAPIExecuteImplicitQuantization(const std::string& model_path, const std::string& quant_file = "",
                                        const std::string& engine_save_path = "/tmp/resnet18_trt_implicit_int8.engine",
                                        nvinfer1::BuilderFlag flag = nvinfer1::BuilderFlag::kINT8);

void IxRTContextMemoryExecute(const std::string& model_path, const std::string& quant_file,
                              const std::string& engine_save_path);

void IxRTContextMemoryExecuteDynamic();

void IxRTAPIExecuteWithHook(const std::string& model_path, const std::string& quant_file = "",
                            const std::string& engine_save_path = "/tmp/resnet18_trt.engine");

void IxRTAPIExecuteFromSerializedONNX();

void IxRTAPIEnqueue(bool use_enqueue_v3 = false);

void IxRTAPIMultiContext();

void IxRTAPIDynamicShape();

void IxRTAPIDynamicShapeMultiContext();

void IxRTAPILoadEngine();

void IxRTAPIExecuteCustomFP32Layers(const std::string& model_path, const std::string& engine_save_path = "",
                                    nvinfer1::BuilderFlag flag = nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);

}  // namespace nvinfer1::samples
