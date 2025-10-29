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
#include <iostream>
#include <memory>

#include "NvInferRuntimeCommon.h"
#include "buffer_utils.h"
#include "cuda_runtime.h"
namespace nvinfer1::samples::common {
template <typename AllocFunc, typename FreeFunc>
class Buffer {
   public:
    using Ptr = std::shared_ptr<Buffer>;
    Buffer(uint64_t size, nvinfer1::DataType data_type) : Buffer() { Resize(size_, data_type); }

    Buffer(const TensorShape& shape) : Buffer() { Resize(shape); }

    ~Buffer() {
        if (buffer_) {
            freeFn(buffer_);
        }
    }

    Buffer() : size_(0), bytes_(0), buffer_(nullptr), data_type_(nvinfer1::DataType::kINT8) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer& operator=(Buffer&& buf) {
        if (this != &buf) {
            if (buffer_) {
                free(buffer_);
            }
            size_ = buf.size_;
            bytes_ = buf.bytes_;
            data_type_ = buf.data_type_;
            buffer_ = buf.buffer_;

            buf.buffer_ = nullptr;
            buf.size_ = 0;
            buf.bytes_ = 0;
        }
        return *this;
    }

    void Resize(uint64_t size, nvinfer1::DataType data_type) {
        size_ = size;
        data_type_ = data_type;
        bytes_ = size * GetDataTypeBytes(data_type);

        if (not allocFn(&buffer_, bytes_)) {
            std::cerr << "Allocate memory for failed with bytes: " << bytes_ << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void Resize(const TensorShape& shape) {
        uint64_t num_element = 1;
        for (auto i = 0; i < shape.dims.size(); ++i) {
            num_element *= shape.dims.at(i) + shape.padding.at(i);
        }
        Resize(num_element, shape.data_type);
    }

    void* GetDataPtr() { return buffer_; }

    uint64_t GetBytes() { return bytes_; }

    uint64_t GetSize() { return size_; }

    nvinfer1::DataType GetDataType() { return data_type_; }

    uint64_t GetDataTypeBytes(nvinfer1::DataType type) {
        switch (type) {
            case nvinfer1::DataType::kINT8:
            case nvinfer1::DataType::kBOOL:
                return 1;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kFLOAT:
            case nvinfer1::DataType::kINT32:
                return 4;
                //            case nvinfer1::DataType::kFLOAT64:
                //                return 8;
            default:
                return 0;
        }
    }

   private:
    void* buffer_;
    uint64_t size_;
    uint64_t bytes_;
    nvinfer1::DataType data_type_;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator {
   public:
    bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

class DeviceFree {
   public:
    void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator {
   public:
    bool operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
   public:
    void operator()(void* ptr) const { free(ptr); }
};

class HostPinnedAllocator {
   public:
    bool operator()(void** ptr, size_t size) const { return cudaMallocHost(ptr, size) == cudaSuccess; }
};

class HostPinnedFree {
   public:
    void operator()(void* ptr) const { cudaFreeHost(ptr); }
};

using DeviceBuffer = Buffer<DeviceAllocator, DeviceFree>;
using HostBuffer = Buffer<HostAllocator, HostFree>;
using HostPinnedBuffer = Buffer<HostPinnedAllocator, HostPinnedFree>;

void SetRandomData(float* data, uint64_t size);

}  // namespace nvinfer1::samples::common
