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
#include <sstream>

#include "NvInfer.h"
#include "NvInferRuntime.h"

// Logs failed assertion and aborts.
// Aborting is undesirable and will be phased-out from the plugin module, at which point
// PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
using namespace std;
#define PLUGIN_ASSERT(value)                                  \
    {                                                         \
        if (not(value)) {                                     \
            std::cerr << __FILE__ << " (" << __LINE__ << ")"  \
                      << "-" << __FUNCTION__ << " : "         \
                      << " Plugin assert false" << std::endl; \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    }

#define PLUGIN_CHECK_CUDA(call)                                             \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

inline void caughtError(const std::exception& e) { std::cerr << e.what() << std::endl; }

#define PLUGIN_FAIL(msg)                              \
    do {                                              \
        std::ostringstream stream;                    \
        stream << "Assertion failed: " << msg << "\n" \
               << __FILE__ << ':' << __LINE__ << "\n" \
               << "Aborting..."                       \
               << "\n";                               \
        PLUGIN_CHECK_CUDA(cudaDeviceReset());         \
        abort;                                        \
    } while (0)
