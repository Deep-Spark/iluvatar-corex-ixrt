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
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>

#include "NvInfer.h"
#undef ASSERT
#define ASSERT(condition)                                                  \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::cerr << "Assertion failure: " << #condition << std::endl; \
            abort();                                                       \
        }                                                                  \
    } while (0)

#ifndef CHECK
#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#endif

namespace nvinfer1::samples::common {
inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
            return 1;
    }
    return 0;
}

inline uint64_t GetBytes(const nvinfer1::Dims& dims, nvinfer1::DataType t) noexcept {
    uint64_t ret{1};
    for (auto i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) {
            ret *= dims.d[i];
        } else {
            std::cerr << "Dim contains dynamic shape" << std::endl;
        }
    }
    if (ret >= 1) {
        return ret * getElementSize(t);
    } else {
        return 0;
    }
}

inline int64_t volume(nvinfer1::Dims const& d) {
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}
inline std::vector<int32_t> Dims2Vec(nvinfer1::Dims const& d) {
    std::vector<int32_t> result;
    result.insert(result.end(), d.d, d.d + d.nbDims);
    return result;
}
inline int getC(nvinfer1::Dims const& d) { return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1; }

inline int getH(const nvinfer1::Dims& d) { return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1; }

inline int getW(const nvinfer1::Dims& d) { return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1; }

inline uint64_t NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace nvinfer1::samples::common
