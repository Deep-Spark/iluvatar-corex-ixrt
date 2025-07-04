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

#include "typed_network.h"
using namespace std;
using namespace nvinfer1::samples;
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "please special infer typed(weakly or strongly)" << std::endl;
        return -1;
    }
    if (argv[1] == std::string("weakly")) {
        WeaklyTypedNetworkSample();
    } else if (argv[1] == std::string("strongly")) {
        StrongTypedNetworkSample();
    } else {
        std::cout << "only weakly or strongly infer type support" << std::endl;
        return -1;
    }
    return 0;
}