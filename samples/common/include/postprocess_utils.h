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
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "data_utils.h"

inline uint64_t _NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
// Single image output image classification
struct ClassificationResult {
    std::vector<float> prob;
    std::vector<int32_t> idx;
    std::vector<std::string> names;
    std::string input_token;
};

struct Box2D {
    float x;
    float y;
    float w;
    float h;
    Box2D(float ix, float iy, float iw, float ih) : x(ix), y(iy), w(iw), h(ih) {}
    Box2D() : Box2D(0.f, 0.f, 0.f, 0.f) {}
    Box2D(const Box2D&) = default;
    Box2D& operator=(const Box2D&) = default;
};

// TODO: Re-design this struct, prototype from qnet
// Single image output detection result
struct DetectionResult {
    Box2D bbox;
    int32_t classes;
    std::vector<float> prob;
    float objectness;
    int32_t sort_class;
    int32_t class_idx;
};

struct DetectionWithLandmark {
    Box2D bbox;
    int32_t classes;
    std::vector<float> prob;
    float objectness;
    int32_t sort_class;
    int32_t class_idx;
    std::vector<float> pts;
};

void RetinafaceGetResults(const IOBuffers& buffers, const IOBuffers& ldm_buffers,
                          std::vector<DetectionWithLandmark>* output, int32_t num_class,
                          const std::vector<int32_t>& num_anchor, int32_t ih, int32_t iw, int32_t batch_idx = 0,
                          float nms_thre = 0.4f, float select_thre = 0.24f);
void PrintRetinafaceDetectionResult(const std::vector<DetectionWithLandmark>& det_outputs, int32_t test_h,
                                    int32_t test_w);
void SaveRetinaFaceDetectionResult(const std::string output_dir, const std::string new_file,
                                   const std::string file_name, const std::vector<DetectionWithLandmark>& det_outputs,
                                   int net_h, int net_w, float scale);

void GetClassificationResult(const float* scores, const int32_t size, int32_t top_k = 5,
                             std::vector<ClassificationResult>* output = nullptr);

void ShowClassificationResults(const std::vector<ClassificationResult>& results);

void PrintDetectionResult(const std::vector<DetectionResult>& det_outputs);

void YoloGetResults(const IOBuffers& buffers, std::vector<DetectionResult>* output, int32_t num_class,
                    const std::vector<int32_t>& num_anchor, int32_t ih, int32_t iw, int32_t batch_idx = 0,
                    float nms_thre = 0.4f, float select_thre = 0.24f);
void YoloGetResults(float* buffer, std::vector<DetectionResult>* output, int32_t num_class, const int32_t num_anchor,
                    int32_t w, int32_t h, int32_t c, int32_t ih, int32_t iw, int32_t batch_idx, float nms_thre = 0.4f,
                    float select_thre = 0.24f);
void YoloV3BatchCellPostProcess(IOBuffers* buffers, void* outputs, int32_t batch_idx, int32_t num_class,
                                std::vector<int32_t> num_anchor, int32_t ih, int32_t iw, float nms_thre = 0.4f,
                                float select_thre = 0.24f);

void PrintDetectionResult(const std::vector<DetectionResult>& det_outputs, int32_t test_h, int32_t test_w);

void YoloV7GetResults(const IOBuffers& buffers, std::vector<DetectionResult>* output, int32_t num_class,
                      const std::vector<int32_t>& num_anchor, int32_t ih, int32_t iw, int32_t batch_idx = 0,
                      float nms_thre = 0.4f, float select_thre = 0.2f);

using DetectionResults = std::vector<DetectionResult>;
using BatchDetectionResults = std::vector<std::shared_ptr<DetectionResults>>;
