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

#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
cudaError_t YoloxDecoderImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, const void* cls_prob,
                             const void* bbox, const void* box_prob, void* output_bbox, void* output_score,
                             int32_t const numClass, int32_t const stride, int32_t const batch, int32_t const height,
                             int32_t const width);
