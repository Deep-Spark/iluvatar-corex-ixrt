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
#include <memory>

#include "cuda_runtime.h"
struct ObjectDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

struct ArrayDeleter {
    template <typename T>
    void operator()(T* p) const {
        delete[] p;
    }
};
template <typename T>
using UniquePtr = std::unique_ptr<T, ObjectDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream) {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
        pStream.reset(nullptr);
    }

    return pStream;
}

static auto EventDeleter = [](cudaEvent_t* p_event) {
    if (p_event) {
        cudaEventDestroy(*p_event);
        delete p_event;
    }
};

inline std::unique_ptr<cudaEvent_t, decltype(EventDeleter)> makeCudaEvent(bool timing = false) {
    std::unique_ptr<cudaEvent_t, decltype(EventDeleter)> p_event(new cudaEvent_t, EventDeleter);
    if (timing) {
        if (cudaEventCreate(p_event.get()) != cudaSuccess) {
            p_event.reset(nullptr);
        }
    } else {
        if (cudaEventCreateWithFlags(p_event.get(), cudaEventDisableTiming) != cudaSuccess) {
            p_event.reset(nullptr);
        }
    }
    return p_event;
}
