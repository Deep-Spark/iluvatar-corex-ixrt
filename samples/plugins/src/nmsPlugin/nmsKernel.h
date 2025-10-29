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
#include <cstdint>

#include "cuda_fp16.h"
#include "driver_types.h"

template <typename T_SCORE>
size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
                                       int topK);

template <typename T>
size_t detectionForwardPreNMSSize(int N, int C2);

template <typename T>
size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

template <typename T_SCORE>
cudaError_t sortScoresPerClass(cudaStream_t stream, const int num, const int num_classes, const int num_preds_per_class,
                               const int background_label_id, const float confidence_threshold, void* conf_scores_gpu,
                               void* index_array_gpu, void* workspace, const int score_bits, const float score_shift);

template <typename T_SCORE>
size_t sortScoresPerClassWorkspaceSize(const int num, const int num_classes, const int num_preds_per_class);

template <typename T_SCORE, typename T_BBOX>
cudaError_t allClassNMS(cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
                        float nms_threshold, bool share_location, bool is_normalized, void* bbox_data,
                        void* before_NMS_scores, void* before_NMS_index_array, void* after_NMS_scores,
                        void* after_NMS_index_array, bool flip_XY, const float score_shift, bool caffe_semantics);

template <typename T_SCORE>
cudaError_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image, void* unsorted_scores,
                               void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices,
                               void* workspace, int score_bits);

template <typename T_SCORE>
size_t sortScoresPerImageWorkspaceSize(const int num_images, const int num_items_per_image);

template <typename T_SCORE, typename T_BBOX>
cudaError_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
                             int numClasses, int topK, int keepTopK, const void* indices, const void* scores,
                             const void* bboxData, void* keepCount, void* nmsedBoxes, void* nmsedScores,
                             void* nmsedClasses, bool clipBoxes, const float scoreShift);
