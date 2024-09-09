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


#include "resnet18.h"

#include <cstdint>
#include <cstdio>
#include <future>

#include "memory_utils.h"
#include "misc.h"

using std::cerr;
using std::cout;
using std::endl;

Resnet18::Resnet18(const std::string& model_path, std::string& quant_file, std::string& input_name,
                   std::string& output_name)
    : input_queue_(1) {
    cout << "call Resnet18 construct" << endl;
    input_queue_flag_.store(false);

    Logger logger(nvinfer1::ILogger::Severity::kWARNING);
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        return;
    } else {
        cout << "Create builder success" << endl;
    }
    if (builder->platformHasFastInt8()) {
        cout << "Current support Int8 inference" << endl;
    } else {
        cout << "Current not support Int8 inference" << endl;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (not network) {
        std::cout << "Create network failed" << std::endl;
        return;
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        return;
    } else {
        cout << "Create onnx parser success" << endl;
    }

    auto num_input = network->getNbInputs();
    cout << "number of input: " << num_input << endl;
    auto num_output = network->getNbOutputs();
    cout << "number of output: " << num_output << endl;

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    ASSERT(inputDims.nbDims == 4);
    cout << "\nInput dimes: " << endl;
    for (auto i = 0; i < inputDims.nbDims; ++i) {
        cout << inputDims.d[i] << " ";
    }

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        return;
    } else {
        cout << "Create serialized engine plan done" << endl;
    }
    // Option operation, not necessary
    // WriteBuffer2Disk("/home/work/trt.engine", plan->data(), plan->size());

    runtime_.reset(nvinfer1::createInferRuntime(logger));

    if (not runtime_) {
        std::cout << "Create runtime failed" << std::endl;
        return;
    } else {
        cout << "Create runtime done" << endl;
    }

    engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    if (not engine_) {
        std::cout << "Create engine failed" << std::endl;
        return;
    } else {
        std::cout << "Create engine done" << endl;
    }

    context_.reset(engine_->createExecutionContext());
    if (context_) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    cout << "Engine name: " << engine_->getName() << endl;
    num_binding_ = engine_->getNbBindings();
    cout << "Number of binding data: " << num_binding_ << endl;
    for (auto i = 0; i < num_binding_; ++i) {
        cout << "The " << i << " binding" << endl;
        cout << "Name: " << engine_->getBindingName(i) << endl;
        cout << "Format: " << (int32_t)engine_->getBindingFormat(i) << endl;
        cout << "Data type: " << (int32_t)engine_->getBindingDataType(i) << endl;
        cout << "Dimension: ";
        for (auto k = 0; k < engine_->getBindingDimensions(i).nbDims; ++k) {
            cout << engine_->getBindingDimensions(i).d[k] << " ";
        }
        cout << endl;
    }
    ASSERT(num_binding_ = 2);
    // output double buffer
    io_buffers_gpu_.resize(engine_->getNbBindings());
    i_buffers_gpu_.resize(1);
    o_double_buffers_gpu_.resize(1 * 2);

    input_idx_ = engine_->getBindingIndex(input_name.c_str());
    cout << "Input index: " << input_idx_ << endl;
    output_idx_ = engine_->getBindingIndex(output_name.c_str());
    cout << "Output index: " << output_idx_ << endl;

    intput_n_volume_ = volume(engine_->getBindingDimensions(input_idx_));
    input_n_bytes_ = GetBytes(engine_->getBindingDimensions(input_idx_), engine_->getBindingDataType(input_idx_));
    output_n_volume_ = volume(engine_->getBindingDimensions(output_idx_));
    output_n_bytes_ = GetBytes(engine_->getBindingDimensions(output_idx_), engine_->getBindingDataType(output_idx_));

    CHECK(cudaMalloc(&i_buffers_gpu_.at(0), input_n_bytes_));
    CHECK(cudaMalloc(&o_double_buffers_gpu_.at(0), output_n_bytes_));
    CHECK(cudaMalloc(&o_double_buffers_gpu_.at(1), output_n_bytes_));
    io_buffers_gpu_.at(input_idx_) = i_buffers_gpu_.at(0);
}

void Resnet18::LoadInput(Tensor* input) {
    input_queue_.Push(input);
    {
        std::unique_lock<std::mutex> locker(input_queue_mtx_);
        input_queue_cv_.wait(locker, [this] { return input_queue_flag_.load(); });
    }
    input_queue_flag_.store(false);
}

void Resnet18::InputDataH2D(int32_t buffer_pos) {
    Tensor* input_cpu_buffer;
    input_queue_.Take(input_cpu_buffer);

    ASSERT(input_cpu_buffer->nb_bytes == input_n_bytes_);

    CHECK(cudaMemcpyAsync(i_buffers_gpu_.at(0), input_cpu_buffer->data, input_cpu_buffer->nb_bytes,
                          cudaMemcpyHostToDevice, h2d_stream_));
    cudaStreamSynchronize(h2d_stream_);
    input_queue_flag_.store(true);
    input_queue_cv_.notify_one();
}

void Resnet18::Infer(int32_t buffer_pos) {
    io_buffers_gpu_.at(output_idx_) = o_double_buffers_gpu_.at(buffer_pos);
    auto status = context_->enqueueV2(io_buffers_gpu_.data(), infer_stream_, &event_);
    if (not status) {
        cout << "Enqueue resnet-18 async failed" << endl;
    }
}

void Resnet18::OutputDataD2H(int32_t buffer_pos) {
    auto output_buffer = std::shared_ptr<float>(new float[output_n_volume_], ArrayDeleter());
    CHECK(cudaMemcpyAsync(output_buffer.get(), o_double_buffers_gpu_.at(buffer_pos), output_n_bytes_,
                          cudaMemcpyDeviceToHost, d2h_stream_));
    cudaStreamSynchronize(d2h_stream_);
    GetClassificationResult(output_buffer.get(), 1000, 5, 0);
}

void Resnet18::SetInputCudaStream(cudaStream_t stream) { h2d_stream_ = stream; }

void Resnet18::SetInferCudaStream(cudaStream_t stream) { infer_stream_ = stream; }

void Resnet18::SetOutputCudaStream(cudaStream_t stream) { d2h_stream_ = stream; }

void Resnet18::SetCudaEvent(cudaEvent_t event) { event_ = event; }
