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

#include <cuda.h>

#include <fstream>

#include "cuda_fp16.h"
#include "yoloxDecodeKernel.h"

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __half sigmoid(__half a) { return __float2half(sigmoid(__half2float(a))); }

template <typename T>
__global__ void YoloxDecoderForward(int32_t const nthreads, T const* cls_prob, T const* bbox, T const* box_prob,
                                    T* output_bbox, T* output_score, int32_t const numClass, int32_t const stride,
                                    int32_t const height, int32_t const width) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    const int32_t hw = height * width;
    const int32_t batchIdx = tid / hw;
    const int32_t pos = tid % hw;
    const int32_t h_idx = pos / width;
    const int32_t w_idx = pos % height;

    const int32_t offset = batchIdx * height * width;
    const int32_t cls_prob_offset = numClass * offset;
    const int32_t bbox_offset = 4 * offset;
    const int32_t box_prob_offset = offset;
    const int32_t out_bbox_offset = bbox_offset;
    const int32_t out_score_offset = cls_prob_offset;

    T xywh0 = bbox[bbox_offset + hw * 0 + pos];
    T xywh1 = bbox[bbox_offset + hw * 1 + pos];
    T xywh2 = bbox[bbox_offset + hw * 2 + pos];
    T xywh3 = bbox[bbox_offset + hw * 3 + pos];

    T bbox_prob = box_prob[box_prob_offset + pos];

    T box_cx = xywh0 + static_cast<T>(w_idx);
    T box_cy = xywh1 + static_cast<T>(h_idx);
    box_cx *= stride;
    box_cy *= stride;

    T box_w = exp(xywh2);
    box_w *= stride;
    T box_h = exp(xywh3);
    box_h *= stride;

    T zero_point_five = static_cast<T>(0.5f);
    T box_left = box_cx - zero_point_five * box_w;
    T box_top = box_cy - zero_point_five * box_h;
    T box_right = box_left + box_w;
    T box_bottom = box_top + box_h;

    output_bbox[out_bbox_offset + hw * 0 + pos] = box_left;
    output_bbox[out_bbox_offset + hw * 1 + pos] = box_top;
    output_bbox[out_bbox_offset + hw * 2 + pos] = box_right;
    output_bbox[out_bbox_offset + hw * 3 + pos] = box_bottom;

    for (int32_t i = 0; i < numClass; i++) {
        T cur_cls_prob = cls_prob[cls_prob_offset + hw * i + pos];
        output_score[out_score_offset + hw * i + pos] = bbox_prob * cur_cls_prob;
    }

    // if (tid<2)
    //     printf("tid %d box_left %f box_top %f box_right %f box_bottom %f class_id 0 bbox_prob * cls_prob %f \n", tid,
    //     (float)output_bbox[out_bbox_offset + hw*0 + pos], (float)output_bbox[out_bbox_offset + hw*1 + pos],
    //     (float)output_bbox[out_bbox_offset + hw*2 + pos], (float)output_bbox[out_bbox_offset + hw*3 + pos],
    //     (float)output_score[out_score_offset + hw * 0 + pos]);
}

template <typename T>
cudaError_t YoloxDecoderImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, const void* cls_prob,
                             const void* bbox, const void* box_prob, void* output_bbox, void* output_score,
                             int32_t const numClass, int32_t const stride, int32_t const batch, int32_t const height,
                             int32_t const width) {
    int32_t const outputSize = batch * height * width;
    int blocksPerGrid = (outputSize + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    int threadsPerBlock = maxThreadsPerBlock;

    YoloxDecoderForward<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        outputSize, static_cast<const T*>(cls_prob), static_cast<const T*>(bbox), static_cast<const T*>(box_prob),
        static_cast<T*>(output_bbox), static_cast<T*>(output_score), numClass, stride, height, width);

    return cudaGetLastError();
}

#define INSTANTIATED_IMPL(T)                                                                                       \
    template cudaError_t YoloxDecoderImpl<T>(                                                                      \
        cudaStream_t stream, int32_t const maxThreadsPerBlock, const void* cls_prob, const void* bbox,             \
        const void* box_prob, void* output_bbox, void* output_score, int32_t const numClass, int32_t const stride, \
        int32_t const batch, int32_t const height, int32_t const width);

INSTANTIATED_IMPL(float)
INSTANTIATED_IMPL(half)
