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

#include <array>
#include <cstddef>

#include "cuda_fp16.h"
#include "nmsKernel.h"

// __half minus with fallback to float for old sm
inline __device__ __half minus_fb(const __half& a, const __half& b) {
    return __float2half(__half2float(a) - __half2float(b));
}

// overload for float
inline __device__ float minus_fb(const float& a, const float& b) { return a - b; }

template <typename T_BBOX>
__device__ T_BBOX saturate(T_BBOX v) {
    return max(min(v, T_BBOX(1)), T_BBOX(0));
}

template <>
__device__ __half saturate(__half v) {
    return max(min(v, float(1)), float(0));
}

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void gatherNMSOutputs_kernel(const bool shareLocation, const int numImages, const int numPredsPerClass,
                                 const int numClasses, const int topK, const int keepTopK, const int* indices,
                                 const T_SCORE* scores, const T_BBOX* bboxData, T_BBOX* numDetections,
                                 T_BBOX* nmsedBoxes, T_BBOX* nmsedScores, T_BBOX* nmsedClasses, bool clipBoxes,
                                 const T_SCORE scoreShift) {
    if (keepTopK > topK) return;
    for (int i = blockIdx.x * nthds_per_cta + threadIdx.x; i < numImages * keepTopK; i += gridDim.x * nthds_per_cta) {
        const int imgId = i / keepTopK;
        const int detId = i % keepTopK;
        const int offset = imgId * numClasses * topK;
        const int index = indices[offset + detId];
        const T_SCORE score = scores[offset + detId];
        if (index == -1) {
            nmsedClasses[i] = -1;
            nmsedScores[i] = 0;
            nmsedBoxes[i * 4] = 0;
            nmsedBoxes[i * 4 + 1] = 0;
            nmsedBoxes[i * 4 + 2] = 0;
            nmsedBoxes[i * 4 + 3] = 0;
        } else {
            // printf("blockIdx.x %d, threadIdx.x %d \n", blockIdx.x, threadIdx.x);
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId =
                ((shareLocation ? (index % numPredsPerClass) : index % (numClasses * numPredsPerClass)) + bboxOffset) *
                4;
            nmsedClasses[i] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass;  // label
            nmsedScores[i] = score;                                                          // confidence score
            nmsedScores[i] = minus_fb(nmsedScores[i], scoreShift);

            int32_t const bbox_data_idx =
                ((shareLocation ? (index % numPredsPerClass) : index % (numClasses * numPredsPerClass)) + bboxOffset);

            // printf("imgId %d, bboxOffset %d, bboxId %d, bbox_data_idx %d \n", imgId, bboxOffset, bboxId,
            // bbox_data_idx);
            if (shareLocation) {
                int32_t const loc_bbox_batch_index = bbox_data_idx / numPredsPerClass;
                int32_t const loc_bbox_in_batch_index = bbox_data_idx % numPredsPerClass;
                int32_t const cur_bbox_data_offset = loc_bbox_batch_index * numPredsPerClass * 4;
                const T_BBOX xMin = bboxData[cur_bbox_data_offset + numPredsPerClass * 0 + loc_bbox_in_batch_index];
                const T_BBOX yMin = bboxData[cur_bbox_data_offset + numPredsPerClass * 1 + loc_bbox_in_batch_index];
                const T_BBOX xMax = bboxData[cur_bbox_data_offset + numPredsPerClass * 2 + loc_bbox_in_batch_index];
                const T_BBOX yMax = bboxData[cur_bbox_data_offset + numPredsPerClass * 3 + loc_bbox_in_batch_index];

                // printf("index %d, loc_bbox_batch_index %d, loc_bbox_in_batch_index %d \n", index,
                // loc_bbox_batch_index, loc_bbox_in_batch_index);

                nmsedBoxes[i * 4] = clipBoxes ? saturate(xMin) : xMin;
                // clipped bbox ymin
                nmsedBoxes[i * 4 + 1] = clipBoxes ? saturate(yMin) : yMin;
                // clipped bbox xmax
                nmsedBoxes[i * 4 + 2] = clipBoxes ? saturate(xMax) : xMax;
                // clipped bbox ymax
                nmsedBoxes[i * 4 + 3] = clipBoxes ? saturate(yMax) : yMax;
                atomicAdd(&numDetections[i / keepTopK], static_cast<T_BBOX>(1.f));
            } else {
                int32_t const loc_bbox_batch_index = bbox_data_idx / (numClasses * numPredsPerClass);
                int32_t const loc_bbox_in_batch_index = bbox_data_idx % (numClasses * numPredsPerClass);
                int32_t const cur_bbox_data_offset = loc_bbox_batch_index * numClasses * numPredsPerClass * 4;
                const T_BBOX xMin =
                    bboxData[cur_bbox_data_offset + numClasses * numPredsPerClass * 0 + loc_bbox_in_batch_index];
                const T_BBOX yMin =
                    bboxData[cur_bbox_data_offset + numClasses * numPredsPerClass * 1 + loc_bbox_in_batch_index];
                const T_BBOX xMax =
                    bboxData[cur_bbox_data_offset + numClasses * numPredsPerClass * 2 + loc_bbox_in_batch_index];
                const T_BBOX yMax =
                    bboxData[cur_bbox_data_offset + numClasses * numPredsPerClass * 3 + loc_bbox_in_batch_index];

                nmsedBoxes[i * 4] = clipBoxes ? saturate(xMin) : xMin;
                // clipped bbox ymin
                nmsedBoxes[i * 4 + 1] = clipBoxes ? saturate(yMin) : yMin;
                // clipped bbox xmax
                nmsedBoxes[i * 4 + 2] = clipBoxes ? saturate(xMax) : xMax;
                // clipped bbox ymax
                nmsedBoxes[i * 4 + 3] = clipBoxes ? saturate(yMax) : yMax;
                atomicAdd(&numDetections[i / keepTopK], static_cast<T_BBOX>(1.f));
            }
        }
    }
}

template <typename T_SCORE, typename T_BBOX>
cudaError_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
                             int numClasses, int topK, int keepTopK, const void* indices, const void* scores,
                             const void* bboxData, void* keepCount, void* nmsedBoxes, void* nmsedScores,
                             void* nmsedClasses, bool clipBoxes, const float scoreShift) {
    cudaMemsetAsync(keepCount, 0, numImages * sizeof(T_SCORE), stream);
    const int BS = 32;
    const int GS = 32;
    gatherNMSOutputs_kernel<T_BBOX, T_SCORE, BS>
        <<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass, numClasses, topK, keepTopK, (int*)indices,
                                (T_SCORE*)scores, (T_BBOX*)bboxData, (T_SCORE*)keepCount, (T_BBOX*)nmsedBoxes,
                                (T_BBOX*)nmsedScores, (T_BBOX*)nmsedClasses, clipBoxes, T_SCORE(scoreShift));

    return cudaGetLastError();
}

#define INSTANTIATED_GATHERNMSOUTPUTS_IMPL(T_SCORE, T_BBOX)                                                     \
    template cudaError_t gatherNMSOutputs<T_SCORE, T_BBOX>(                                                     \
        cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass, int numClasses, int topK, \
        int keepTopK, const void* indices, const void* scores, const void* bboxData, void* keepCount,           \
        void* nmsedBoxes, void* nmsedScores, void* nmsedClasses, bool clipBoxes, const float scoreShift);

INSTANTIATED_GATHERNMSOUTPUTS_IMPL(float, float)
INSTANTIATED_GATHERNMSOUTPUTS_IMPL(half, half)
