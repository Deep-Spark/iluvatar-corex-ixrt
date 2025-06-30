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

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace nvinfer1 {

namespace plugin {

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val) {
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer) {
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

}  // namespace plugin

}  // namespace nvinfer1
