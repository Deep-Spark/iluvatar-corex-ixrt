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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <ostream>
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "driver_types.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "postprocess_utils.h"
#include "sync_queue.h"

using std::cerr;
using std::cout;
using std::endl;

struct Tensor {
    using Ptr = std::shared_ptr<Tensor>;
    void* data;
    size_t nb_bytes;
    Tensor() : data(nullptr) {}
    Tensor(void* d, size_t nb_b) : data(d), nb_bytes(nb_b) {}
};

class Resnet18 {
   public:
    using Ptr = std::shared_ptr<Resnet18>;
    Resnet18(const std::string& model_path, std::string& quant_file, std::string& input_name, std::string& output_name);
    ~Resnet18() { destroy(); };

    void LoadInput(Tensor* input);
    void InputDataH2D(int32_t buffer_pos);
    void Infer(int32_t buffer_pos);
    void OutputDataD2H(int32_t buffer_pos);
    void SetInputCudaStream(cudaStream_t stream);
    void SetInferCudaStream(cudaStream_t stream);
    void SetOutputCudaStream(cudaStream_t stream);
    void SetCudaEvent(cudaEvent_t event);

   private:
    void destroy() {
        context_ = nullptr;
        engine_ = nullptr;
        runtime_ = nullptr;
        for (void* ptr : i_buffers_gpu_) cudaFree(ptr);
        for (void* ptr : o_double_buffers_gpu_) cudaFree(ptr);
    }

   public:
    int32_t num_binding_;
    int32_t input_idx_;
    int32_t output_idx_;
    int32_t intput_n_volume_;
    int32_t input_n_bytes_;
    int32_t output_n_volume_;
    int32_t output_n_bytes_;

    UniquePtr<nvinfer1::IExecutionContext> context_;
    UniquePtr<nvinfer1::ICudaEngine> engine_;
    UniquePtr<nvinfer1::IRuntime> runtime_;

    std::vector<void*> io_buffers_gpu_;
    std::vector<void*> i_buffers_gpu_;
    std::vector<void*> o_double_buffers_gpu_;

    cudaStream_t h2d_stream_ = nullptr;
    cudaStream_t infer_stream_ = nullptr;
    cudaStream_t d2h_stream_ = nullptr;
    cudaEvent_t event_ = nullptr;

    SyncQueue<Tensor*> input_queue_;

    std::mutex input_queue_mtx_;
    std::atomic<bool> input_queue_flag_;
    std::condition_variable input_queue_cv_;
    Logger logger_;
};
