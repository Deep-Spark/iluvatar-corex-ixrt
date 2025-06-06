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

#include "quantization.h"

using namespace std;

void EchoHelp() {}

void RunExample(std::string choice) {
    if (choice == "tex_quant") {
        IxRTAPIQuant("data/resnet50/resnet50.onnx");
    } else if (choice == "-h" or choice == "--help" or choice == "help") {
        EchoHelp();
    } else {
        std::cout << "No option running:" << choice << std::endl;
        EchoHelp();
    }
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        RunExample(argv[i]);
    }
    return 0;
}
