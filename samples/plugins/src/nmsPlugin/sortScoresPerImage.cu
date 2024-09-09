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

#include "cub/cub.cuh"
#include "cubHelper.h"
#include "cuda_fp16.h"
#include "nmsKernel.h"

template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void setUniformOffsets_kernel(const int num_segments, const int offset, int* d_offsets) {
    const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    if (idx <= num_segments) d_offsets[idx] = idx * offset;
}

void setUniformOffsets(cudaStream_t stream, const int num_segments, const int offset, int* d_offsets) {
    const int BS = 32;
    const int GS = (num_segments + 1 + BS - 1) / BS;
    setUniformOffsets_kernel<BS><<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}

template <typename T_SCORE>
cudaError_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image, void* unsorted_scores,
                               void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices,
                               void* workspace, int score_bits) {
    void* d_offsets = workspace;
    void* cubWorkspace = nextWorkspacePtr((int8_t*)d_offsets, (num_images + 1) * sizeof(int));

    setUniformOffsets(stream, num_images, num_items_per_image, (int*)d_offsets);

    const int arrayLen = num_images * num_items_per_image;
    size_t temp_storage_bytes = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_images);
    size_t begin_bit = 0;
    size_t end_bit = sizeof(T_SCORE) * 8;
    if (sizeof(T_SCORE) == 2 && score_bits > 0 && score_bits <= 10) {
        end_bit = 10;
        begin_bit = end_bit - score_bits;
    }
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cubWorkspace, temp_storage_bytes, (const T_SCORE*)(unsorted_scores), (T_SCORE*)(sorted_scores),
        (const int*)(unsorted_bbox_indices), (int*)(sorted_bbox_indices), arrayLen, num_images, (const int*)d_offsets,
        (const int*)d_offsets + 1, begin_bit, end_bit, stream);
    return cudaGetLastError();
}

#define INSTANTIATED_SORTSCORESPERIMAGE_IMPL(T_SCORE)                                        \
    template cudaError_t sortScoresPerImage<T_SCORE>(                                        \
        cudaStream_t stream, int num_images, int num_items_per_image, void* unsorted_scores, \
        void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices, void* workspace, int score_bits);

INSTANTIATED_SORTSCORESPERIMAGE_IMPL(float)
INSTANTIATED_SORTSCORESPERIMAGE_IMPL(half)

template <typename T_SCORE>
size_t sortScoresPerImageWorkspaceSize(const int num_images, const int num_items_per_image) {
    const int arrayLen = num_images * num_items_per_image;
    size_t wss[2];
    wss[0] = (num_images + 1) * sizeof(int);                                 // offsets
    wss[1] = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_images);  // cub workspace

    return calculateTotalWorkspaceSize(wss, 2);
}

#define INSTANTIATED_SORTSCORESPERIMAGEWS_IMPL(T_SCORE) \
    template size_t sortScoresPerImageWorkspaceSize<T_SCORE>(const int num_images, const int num_items_per_image);

INSTANTIATED_SORTSCORESPERIMAGEWS_IMPL(float)
INSTANTIATED_SORTSCORESPERIMAGEWS_IMPL(half)
