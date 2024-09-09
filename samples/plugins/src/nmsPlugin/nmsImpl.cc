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


#include "nmsImpl.h"

#include <cstdio>

#include "PluginCheckMacros.h"
#include "nmsKernel.h"

template <typename T>
cudaError_t NMSImpl(cudaStream_t stream, int32_t const max_threads_per_block, const int batch_size,
                    const int per_batch_boxes_size, const int per_batch_scores_size, const bool share_location,
                    const int background_label_id, const int num_priors, const int num_classes, const int keep_topk,
                    const float score_threshold, const float iou_threshold, const void* boxes_data,
                    const void* scores_data, void* keep_count, void* nmsed_boxes, void* nmsed_scores,
                    void* nmsed_classes, void* workspace) {
    cudaError_t statue;

    size_t indices_size = detectionForwardPreNMSSize<int>(batch_size, per_batch_scores_size);
    void* indices = workspace;

    size_t post_NMS_scores_size = detectionForwardPostNMSSize<T>(batch_size, num_classes, num_priors);
    void* post_NMS_scores = nextWorkspacePtr((int8_t*)indices, indices_size);
    size_t post_NMS_indices_size =
        detectionForwardPostNMSSize<int>(batch_size, num_classes, num_priors);  // indices are full int32
    void* post_NMS_indices = nextWorkspacePtr((int8_t*)post_NMS_scores, post_NMS_scores_size);

    void* sorting_workspace = nextWorkspacePtr((int8_t*)post_NMS_indices, post_NMS_indices_size);
    // Sort the scores so that the following NMS could be applied.
    int score_bits = 16;
    float score_shift = 0.f;
    if (sizeof(T) == 2 && score_bits > 0 && score_bits <= 10) score_shift = 1.f;

    statue = sortScoresPerClass<T>(stream, batch_size, num_classes, num_priors, background_label_id, score_threshold,
                                   const_cast<void*>(scores_data), indices, sorting_workspace, score_bits, score_shift);
    PLUGIN_ASSERT(statue == cudaSuccess);

    // NMS
    bool is_normalized = false;
    bool flip_XY = false;
    bool caffe_semantics = false;
    int top_k = 4096;
    statue = allClassNMS<T, T>(stream, batch_size, num_classes, num_priors, top_k, iou_threshold, share_location,
                               is_normalized, const_cast<void*>(boxes_data), const_cast<void*>(scores_data), indices,
                               post_NMS_scores, post_NMS_indices, flip_XY, score_shift, caffe_semantics);
    PLUGIN_ASSERT(statue == cudaSuccess);

    // Sort the bounding boxes after NMS using scores
    statue = sortScoresPerImage<T>(stream, batch_size, num_classes * top_k, post_NMS_scores, post_NMS_indices,
                                   const_cast<void*>(scores_data), indices, sorting_workspace, score_bits);
    PLUGIN_ASSERT(statue == cudaSuccess);

    // Gather data from the sorted bounding boxes after NMS
    bool clip_boxes = false;
    statue = gatherNMSOutputs<T, T>(stream, share_location, batch_size, num_priors, num_classes, top_k, keep_topk,
                                    indices, scores_data, boxes_data, keep_count, nmsed_boxes, nmsed_scores,
                                    nmsed_classes, clip_boxes, score_shift);
    PLUGIN_ASSERT(statue == cudaSuccess);

    return cudaGetLastError();
}

#define INSTANTIATED_IMPL(T)                                                                                     \
    template cudaError_t NMSImpl<T>(                                                                             \
        cudaStream_t stream, int32_t const max_threads_per_block, const int batch_size,                          \
        const int per_batch_boxes_size, const int per_batch_scores_size, const bool share_location,              \
        const int background_label_id, const int num_priors, const int num_classes, const int keep_topk,         \
        const float score_threshold, const float iou_threshold, const void* boxes_data, const void* scores_data, \
        void* keep_count, void* nmsed_boxes, void* nmsed_scores, void* nmsed_classes, void* workspace);

INSTANTIATED_IMPL(float)
INSTANTIATED_IMPL(half)
