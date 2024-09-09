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
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"

namespace nvinfer1::samples::common {
struct TensorShape {
    std::vector<int32_t> dims;
    std::vector<int32_t> padding;
    nvinfer1::DataType data_type;
    TensorFormat data_format;
    TensorShape(const std::vector<int32_t>& in_dims, const std::vector<int32_t>& in_pad, nvinfer1::DataType in_dtype,
                TensorFormat in_dformat)
        : dims(in_dims), padding(in_pad), data_type(in_dtype), data_format(in_dformat) {}
    TensorShape() : data_type(nvinfer1::DataType::kINT8), data_format(TensorFormat::kLINEAR) {}
    TensorShape(const TensorShape&) = default;
    TensorShape& operator=(const TensorShape&) = default;
};

struct IOBuffer {
    using Ptr = std::shared_ptr<IOBuffer>;
    std::string name;
    void* data;
    TensorShape shape;
    IOBuffer() : data(nullptr) {}
    IOBuffer(const std::string& n, void* d, const TensorShape& s = TensorShape()) : name(n), data(d), shape(s) {}
};
enum AlgoSelectMode { ALGO_SELECT_MODE_MANUAL = 0, ALGO_SELECT_MODE_AUTO_TIMEING };
using TensorShapeMap = std::unordered_map<std::string, TensorShape>;
using IOBuffers = std::vector<IOBuffer>;

}  // namespace nvinfer1::samples::common
