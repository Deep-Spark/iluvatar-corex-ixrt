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
cudaError_t NMSImpl(cudaStream_t stream, int32_t const max_threads_per_block, const int batch_size,
                    const int per_batch_boxes_size, const int per_batch_scores_size, const bool share_location,
                    const int background_label_id, const int num_priors, const int num_classes, const int keep_topk,
                    const float score_threshold, const float iou_threshold, const void* boxes_data,
                    const void* scores_data, void* keep_count, void* nmsed_boxes, void* nmsed_scores,
                    void* nmsed_classes, void* workspace);
