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




#include "pipeline.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>

Pipeline::Pipeline() : shut_down_(true), device_id_(0) {
    cudaStreamCreateWithFlags(&input_stream_, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&infer_stream_, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&output_stream_, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}

Pipeline::~Pipeline() {
    worker_ = nullptr;
    cudaStreamDestroy(input_stream_);
    cudaStreamDestroy(infer_stream_);
    cudaStreamDestroy(output_stream_);
    cudaEventDestroy(event_);
}

void Pipeline::ResetFlag() {
    input_data_readable_.store(false);
    input_data_consumed_.store(true);
    output_data_0_writable_.store(true);
    output_data_0_produced_.store(false);
    output_data_1_writable_.store(true);
    output_data_1_produced_.store(false);
    shut_down_.store(true);
}

void Pipeline::Start(const WorkerLoadMethod& worker_loadmethod) {
    worker_.reset(worker_loadmethod());
    SetTransferInput(std::bind(&Resnet18::InputDataH2D, worker_, std::placeholders::_1));
    SetInfer(std::bind(&Resnet18::Infer, worker_, std::placeholders::_1));
    SetTransferOutput(std::bind(&Resnet18::OutputDataD2H, worker_, std::placeholders::_1));

    worker_->SetInputCudaStream(GetInputCudaStream());
    worker_->SetInferCudaStream(GetInferCudaStream());
    worker_->SetOutputCudaStream(GetOutputCudaStream());
    worker_->SetCudaEvent(GetCudaEvent());

    ResetFlag();
    shut_down_.store(false);
    h2d_thread_ptr_ = new std::thread(&Pipeline::TransferInput, this);
    infer_thread_ptr_ = new std::thread(&Pipeline::Infer, this);
    d2h_0_thread_ptr_ = new std::thread(&Pipeline::TransferOutput, this, 0);
    d2h_1_thread_ptr_ = new std::thread(&Pipeline::TransferOutput, this, 1);
}

void Pipeline::Stop() {
    shut_down_.store(true);

    input_data_consumed_cv_.notify_one();
    runable_cv_.notify_one();
    output_data_0_produced_cv_.notify_one();
    output_data_1_produced_cv_.notify_one();

    h2d_thread_ptr_->join();
    infer_thread_ptr_->join();
    d2h_0_thread_ptr_->join();
    d2h_1_thread_ptr_->join();

    delete h2d_thread_ptr_;
    delete infer_thread_ptr_;
    delete d2h_0_thread_ptr_;
    delete d2h_1_thread_ptr_;
}

cudaStream_t Pipeline::GetInputCudaStream() { return input_stream_; }

cudaStream_t Pipeline::GetInferCudaStream() { return infer_stream_; }

cudaStream_t Pipeline::GetOutputCudaStream() { return output_stream_; }

cudaEvent_t Pipeline::GetCudaEvent() { return event_; }

void Pipeline::SetDeviceID(uint32_t device_id) { device_id_ = device_id; }

void Pipeline::SetTransferInput(const TaskMethod& func) { transfer_input_task_ = func; }

void Pipeline::SetInfer(const TaskMethod& func) { infer_task_ = func; }

void Pipeline::SetTransferOutput(const TaskMethod& func) { transfer_output_task_ = func; }

void Pipeline::CommitInput(Tensor* input) { worker_->LoadInput(input); }

void Pipeline::TransferInput() {
    CHECK(cudaSetDevice(device_id_));
    int32_t input_buffer_pos = 0;
    while (true) {
        {
            std::unique_lock<std::mutex> locker(input_consumer_mtx_);
            input_data_consumed_cv_.wait(locker, [this] { return shut_down_.load() || input_data_consumed_.load(); });
        }
        if (shut_down_.load()) {
            break;
        }
        transfer_input_task_(input_buffer_pos);

        input_data_readable_.store(true);
        input_data_consumed_.store(false);
        runable_cv_.notify_one();
    }
}

void Pipeline::Infer() {
    CHECK(cudaSetDevice(device_id_));
    int32_t output_buffer_pos = 0;
    while (true) {
        if (output_buffer_pos == 0) {
            {
                std::unique_lock<std::mutex> locker(infer_mtx_);
                runable_cv_.wait(locker, [this] {
                    return shut_down_.load() || (input_data_readable_.load() && output_data_0_writable_.load());
                });
            }
            if (shut_down_.load()) {
                break;
            }
            infer_task_(output_buffer_pos);
            cudaEventSynchronize(event_);
            input_data_readable_.store(false);
            input_data_consumed_.store(true);
            input_data_consumed_cv_.notify_one();

            cudaStreamSynchronize(infer_stream_);
            output_data_0_writable_.store(false);
            output_data_0_produced_.store(true);
            output_data_0_produced_cv_.notify_one();
        } else {
            {
                std::unique_lock<std::mutex> locker(infer_mtx_);
                runable_cv_.wait(locker, [this] {
                    return shut_down_.load() || (input_data_readable_.load() && output_data_1_writable_.load());
                });
            }
            if (shut_down_.load()) {
                break;
            }
            infer_task_(output_buffer_pos);
            cudaEventSynchronize(event_);
            input_data_readable_.store(false);
            input_data_consumed_.store(true);
            input_data_consumed_cv_.notify_one();

            cudaStreamSynchronize(infer_stream_);
            output_data_1_writable_.store(false);
            output_data_1_produced_.store(true);
            output_data_1_produced_cv_.notify_one();
        }
        output_buffer_pos ^= 1;
    }
}

void Pipeline::TransferOutput(int32_t buffer_pos) {
    CHECK(cudaSetDevice(device_id_));
    if (buffer_pos == 0) {
        while (true) {
            {
                std::unique_lock<std::mutex> locker(output_producter_mtx_);
                output_data_0_produced_cv_.wait(locker,
                                                [this] { return shut_down_ || output_data_0_produced_.load(); });
            }
            if (shut_down_.load()) {
                break;
            }
            transfer_output_task_(buffer_pos);
            output_data_0_writable_.store(true);
            output_data_0_produced_.store(false);
            runable_cv_.notify_one();
        }
    } else {
        while (true) {
            {
                std::unique_lock<std::mutex> locker(output_producter_mtx_);
                output_data_1_produced_cv_.wait(locker,
                                                [this] { return shut_down_ || output_data_1_produced_.load(); });
            }
            if (shut_down_.load()) {
                break;
            }
            transfer_output_task_(buffer_pos);
            output_data_1_writable_.store(true);
            output_data_1_produced_.store(false);
            runable_cv_.notify_one();
        }
    }
}
