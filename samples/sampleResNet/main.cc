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

#include <iostream>
#include <string>

#include "classification.h"

using namespace std;
using namespace nvinfer1::samples;
void EchoHelp() {
    std::string help_text =
        "------------------ResNet18 simple example------------------\n"
        "How to run                                                 \n"
        "   ./sampleResNet TARGET                                      \n"
        "   where TARGET can be:                                    \n"
        "  - tex_i8_explicit                                         \n"
        "  - tex_i8_implicit                                         \n"
        "  - tex_fp16                                                \n"
        "  - tex_fp32                                                \n"
        "  - tex_s_onnx                                              \n"
        "  - ten                                                     \n"
        "  - tmc                                                     \n"
        "  - ted                                                     \n"
        "  - tmcd                                                    \n"
        "  - load_engine                                             \n"
        "  - hook                                                    \n"
        "-----------------------------------------------------------\n";
    std::cout << help_text << std::endl;
}
void RunExample(std::string choice) {
    if (choice == "tex_i8_explicit") {
        IxRTAPIExecute("data/resnet18/resnet18_qdq_int8.onnx", "", "/tmp/resnet18_i8.engine",
                       nvinfer1::BuilderFlag::kINT8);
    } else if (choice == "tex_i8_implicit") {
        IxRTAPIExecuteImplicitQuantization("data/resnet18/resnet18.onnx",
                                           "data/resnet18/resnet18_int8_quantization.json",
                                           "/tmp/resnet18_i8_implicit.engine", nvinfer1::BuilderFlag::kINT8);
    } else if (choice == "tex_fp16") {
        IxRTAPIExecute("data/resnet18/resnet18.onnx", "", "/tmp/resnet18_fp16.engine", nvinfer1::BuilderFlag::kFP16);
    } else if (choice == "tex_fp32") {
        IxRTAPIExecuteCustomFP32Layers("data/resnet18/resnet18.onnx", "",
                                       nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    } else if (choice == "tex_s_onnx") {
        IxRTAPIExecuteFromSerializedONNX();
    } else if (choice == "ten") {
        IxRTAPIEnqueue();
    } else if (choice == "tmc") {
        IxRTAPIMultiContext();
    } else if (choice == "ted") {
        IxRTAPIDynamicShape();
    } else if (choice == "tmcd") {
        IxRTAPIDynamicShapeMultiContext();
    } else if (choice == "load_engine") {
        IxRTAPILoadEngine();
    } else if (choice == "hook") {
        IxRTAPIExecuteWithHook("data/resnet18/resnet18.onnx");
    } else if (choice == "txt_mem") {
        IxRTContextMemoryExecute("data/resnet18/resnet18.onnx", "", "/tmp/resnet18_fp16.engine");
    } else if (choice == "txt_memd") {
        IxRTContextMemoryExecuteDynamic();
    } else if (choice == "-h" or choice == "--help" or choice == "help") {
        EchoHelp();
    } else {
        std::cout << "No option running:" << choice << std::endl;
        EchoHelp();
    }
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        RunExample("execute");
    }
    for (int i = 1; i < argc; ++i) {
        RunExample(argv[i]);
    }
    return 0;
}
