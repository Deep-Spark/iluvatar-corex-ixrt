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


#include <cuda_runtime_api.h>

#include "PluginCheckMacros.h"
#include "nmsKernel.h"
#define CUDA_MEM_ALIGN 256

template <typename T_SCORE>
size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
                                       int topK) {
    size_t wss[4];
    wss[0] = detectionForwardPreNMSSize<int>(N, C2);
    wss[1] = detectionForwardPostNMSSize<T_SCORE>(N, numClasses, topK);
    wss[2] = detectionForwardPostNMSSize<int>(N, numClasses, topK);
    wss[3] = std::max(sortScoresPerClassWorkspaceSize<T_SCORE>(N, numClasses, numPredsPerClass),
                      sortScoresPerImageWorkspaceSize<T_SCORE>(N, numClasses * topK));
    return calculateTotalWorkspaceSize(wss, 4);
}

#define INSTANTIATED_DETECTIONINFERENCEWSSIZE(T)                                                                  \
    template size_t detectionInferenceWorkspaceSize<T>(bool shareLocation, int N, int C1, int C2, int numClasses, \
                                                       int numPredsPerClass, int topK);

INSTANTIATED_DETECTIONINFERENCEWSSIZE(float)
INSTANTIATED_DETECTIONINFERENCEWSSIZE(half)

template <typename T>
size_t detectionForwardPreNMSSize(int N, int C2) {
    PLUGIN_ASSERT(sizeof(float) == sizeof(int));
    return N * C2 * sizeof(T);
}

#define INSTANTIATED_DETECTIONFORWARDPRENMSSIZE(T) template size_t detectionForwardPreNMSSize<T>(int N, int C2);

INSTANTIATED_DETECTIONFORWARDPRENMSSIZE(int)

template <typename T>
size_t detectionForwardPostNMSSize(int N, int numClasses, int topK) {
    PLUGIN_ASSERT(sizeof(float) == sizeof(int));
    return N * numClasses * topK * sizeof(T);
}

#define INSTANTIATED_DETECTIONFORWARDPOSTNMSSIZE(T) \
    template size_t detectionForwardPostNMSSize<T>(int N, int numClasses, int topK);

INSTANTIATED_DETECTIONFORWARDPOSTNMSSIZE(int)
INSTANTIATED_DETECTIONFORWARDPOSTNMSSIZE(float)
INSTANTIATED_DETECTIONFORWARDPOSTNMSSIZE(half)

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t*)addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*)addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count) {
    size_t total = 0;
    for (int i = 0; i < count; i++) {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN) {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}
