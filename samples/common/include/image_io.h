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
#include <vector>

#include "NvInfer.h"
#include "buffer_utils.h"
#include "data_utils.h"
namespace nvinfer1::samples::common {
struct Image {
    int32_t h;
    int32_t w;
    int32_t c;
    HostBuffer* data;
    Image() : h(0), w(0), c(0), data(nullptr) {}
};

std::shared_ptr<uint8_t> GetImagePtr(const std::string& file_name, int32_t* w, int32_t* h, int32_t* c, int32_t mode);

template <typename BufferType>
void LoadImageCPU(const std::string& file_name, BufferType* image, const std::vector<int32_t>& tensor_shape,
                  int32_t batch_idx = 0, bool use_hwc = false,
                  const std::vector<float>& mean = std::vector<float>{0.f, 0.f, 0.f},
                  const std::vector<float>& std = std::vector<float>{255.f, 255.f, 255.f}, bool convert_to_bgr = false);

template <typename BufferType>
void LoadImageCPUYolox(const std::string& file_name, BufferType* image, const TensorShape& tensor_shape,
                       int32_t batch_idx = 0);

template <typename BufferType>
void LoadImageCPURetinaFace(const std::string& file_name, BufferType* image, const TensorShape& tensor_shape,
                            float* means, float* stds, int32_t batch_idx = 0);

std::shared_ptr<float> LoadImageCPU(const std::string& file_name, const nvinfer1::Dims& dims, bool chw = true);
std::shared_ptr<float> LoadNormalizeImageCPU(const std::string& file_name, const nvinfer1::Dims& dims,
                                             const std::vector<float>& mean = std::vector<float>{0.485f, 0.456f,
                                                                                                 0.406f},
                                             const std::vector<float>& std = std::vector<float>{0.229f, 0.224f,
                                                                                                0.225f});
void LoadImageBuffer(const std::string& file_name, const nvinfer1::Dims& dims, float* data, int32_t batch = 0);

}  // namespace nvinfer1::samples::common
