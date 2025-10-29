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
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

#include "NvInferRuntimeCommon.h"

namespace nvinfer1::samples::common {
class SampleErrorRecorder : public IErrorRecorder {
    using errorPair = std::pair<ErrorCode, std::string>;
    using errorStack = std::vector<errorPair>;

   public:
    SampleErrorRecorder() = default;

    ~SampleErrorRecorder() noexcept override {}
    int32_t getNbErrors() const noexcept final { return mErrorStack.size(); }
    ErrorCode getErrorCode(int32_t errorIdx) const noexcept final {
        return invalidIndexCheck(errorIdx) ? ErrorCode::kINVALID_ARGUMENT : (*this)[errorIdx].first;
    };
    IErrorRecorder::ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept final {
        return invalidIndexCheck(errorIdx) ? "errorIdx out of range." : (*this)[errorIdx].second.c_str();
    }
    bool hasOverflowed() const noexcept final { return false; }

    void clear() noexcept final {
        try {
            std::lock_guard<std::mutex> guard(mStackLock);
            mErrorStack.clear();
        } catch (const std::exception& e) {
            throw e.what();
        }
    };

    bool empty() const noexcept { return mErrorStack.empty(); }

    bool reportError(ErrorCode val, IErrorRecorder::ErrorDesc desc) noexcept final {
        try {
            std::lock_guard<std::mutex> guard(mStackLock);
            std::cerr << "Error[" << static_cast<int32_t>(val) << "]: " << desc << std::endl;
            mErrorStack.push_back(errorPair(val, desc));
        } catch (const std::exception& e) {
            throw e.what();
        }
        return true;
    }

    IErrorRecorder::RefCount incRefCount() noexcept final { return ++mRefCount; }
    IErrorRecorder::RefCount decRefCount() noexcept final { return --mRefCount; }

   private:
    const errorPair& operator[](size_t index) const noexcept { return mErrorStack[index]; }

    bool invalidIndexCheck(int32_t index) const noexcept {
        size_t sIndex = index;
        return sIndex >= mErrorStack.size();
    }
    std::mutex mStackLock;

    std::atomic<int32_t> mRefCount{0};

    errorStack mErrorStack;
};  // class SampleErrorRecorder

}  // namespace nvinfer1::samples::common
