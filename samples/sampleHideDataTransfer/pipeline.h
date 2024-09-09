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
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "cuda_runtime.h"
#include "misc.h"
#include "resnet18.h"

using TaskMethod = std::function<void(int32_t)>;
using WorkerLoadMethod = std::function<Resnet18*()>;
class Pipeline {
   public:
    using Ptr = std::shared_ptr<Pipeline>;
    Pipeline();
    ~Pipeline();
    void Start(const WorkerLoadMethod& loadmethod);
    void Stop();
    void SetDeviceID(uint32_t device_id = 0);
    void CommitInput(Tensor* input);

   private:
    void SetTransferInput(const TaskMethod& func);
    void SetInfer(const TaskMethod& func);
    void SetTransferOutput(const TaskMethod& func);

    cudaStream_t GetInputCudaStream();
    cudaStream_t GetInferCudaStream();
    cudaStream_t GetOutputCudaStream();
    cudaEvent_t GetCudaEvent();

    void TransferInput();
    void Infer();
    void TransferOutput(int32_t buffer_pos);
    void ResetFlag();

    cudaStream_t input_stream_;
    cudaStream_t infer_stream_;
    cudaStream_t output_stream_;
    cudaEvent_t event_;

    std::atomic<bool> input_data_readable_;
    std::atomic<bool> input_data_consumed_;
    // output double buffer
    std::atomic<bool> output_data_0_writable_;
    std::atomic<bool> output_data_0_produced_;
    std::atomic<bool> output_data_1_writable_;
    std::atomic<bool> output_data_1_produced_;
    std::atomic<bool> shut_down_;

    std::condition_variable runable_cv_;
    std::condition_variable input_data_consumed_cv_;
    std::condition_variable output_data_0_produced_cv_;
    std::condition_variable output_data_1_produced_cv_;

    std::mutex infer_mtx_;
    std::mutex input_consumer_mtx_;
    std::mutex output_producter_mtx_;

    TaskMethod transfer_input_task_;
    TaskMethod infer_task_;
    TaskMethod transfer_output_task_;

    uint32_t device_id_;

    // sub thread
    std::thread* h2d_thread_ptr_;
    std::thread* infer_thread_ptr_;
    std::thread* d2h_0_thread_ptr_;
    std::thread* d2h_1_thread_ptr_;

    // worker
    Resnet18::Ptr worker_ = nullptr;
};
