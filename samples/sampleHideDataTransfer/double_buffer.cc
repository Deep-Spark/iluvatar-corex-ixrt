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


#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "hide_data_transfer.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "postprocess_utils.h"
using std::cerr;
using std::cout;
using std::endl;

void IxRTAPIEnqueueHideDataTransferDoubleBuffer() {
    std::string dir_path("data/resnet18/");
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string model_path(dir_path + "resnet18.onnx");
    std::string input_name("input");
    std::string output_name("output");
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

    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        return;
    } else {
        cout << "Create runtime done" << endl;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        return;
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbBindings();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;
        cout << "Name: " << engine->getBindingName(i) << endl;
        cout << "Format: " << (int32_t)engine->getBindingFormat(i) << endl;
        cout << "Data type: " << (int32_t)engine->getBindingDataType(i) << endl;
        cout << "Dimension: ";
        for (auto k = 0; k < engine->getBindingDimensions(i).nbDims; ++k) {
            cout << engine->getBindingDimensions(i).d[k] << " ";
        }
        cout << endl;
    }

    auto input_idx = engine->getBindingIndex(input_name.c_str());
    cout << "Input index: " << input_idx << endl;
    auto output_idx = engine->getBindingIndex(output_name.c_str());
    cout << "Output index: " << output_idx << endl;

    std::vector<void*> warmup_io_buffers_gpu(num_bd);
    // double buffer
    std::vector<void*> io_buffers_gpu(num_bd * 2);

    int intput_n_volume;
    int input_n_bytes;
    int output_n_volume;
    int output_n_bytes;
    // malloc device memery
    for (int i = 0; i < num_bd; ++i) {
        int n_volume = volume(engine->getBindingDimensions(i));
        int n_bytes = GetBytes(engine->getBindingDimensions(i), engine->getBindingDataType(i));
        if (i == input_idx) {
            intput_n_volume = n_volume;
            input_n_bytes = n_bytes;
        } else if (i == output_idx) {
            output_n_volume = n_volume;
            output_n_bytes = n_bytes;
        }
        CHECK(cudaMalloc(&warmup_io_buffers_gpu.at(i), n_bytes));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i), n_bytes));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i + num_bd), n_bytes));
    }

    std::queue<std::shared_ptr<float>> input_buffers_queue;
    std::queue<std::shared_ptr<float>> output_buffers_queue;

    auto input_dims = Dims2Vec(engine->getBindingDimensions(input_idx));
    std::shared_ptr<float> input_buffer;
    float* tmp_input_ptr;
    CHECK(cudaMallocHost(&tmp_input_ptr, intput_n_volume * sizeof(float)));
    input_buffer.reset(tmp_input_ptr, [](float* p) { CHECK(cudaFreeHost(p)); });
    LoadImageCPU(image_path, input_buffer.get(), input_dims, 0);
    for (int i = 0; i < 20; i++) {
        input_buffers_queue.push(input_buffer);
    }
    cout << "User input date queue prepare done" << endl;

    auto context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto h2d_stream_ptr = makeCudaStream();
    auto infer_stream_ptr = makeCudaStream();
    auto d2h_stream_ptr = makeCudaStream();
    CHECK(cudaMemcpyAsync(warmup_io_buffers_gpu.at(input_idx), input_buffers_queue.front().get(), input_n_bytes,
                          cudaMemcpyHostToDevice, *h2d_stream_ptr));
    cudaStreamSynchronize(*h2d_stream_ptr);
    auto warmup_output_buffer = std::shared_ptr<float>(new float[output_n_volume], ArrayDeleter());
    for (auto i = 0; i < 20; ++i) {
        context->enqueueV2(warmup_io_buffers_gpu.data(), *infer_stream_ptr, nullptr);
    }
    cudaStreamSynchronize(*infer_stream_ptr);
    std::cout << "Warm up done" << endl;

    int io_buffer_pos = 0;
    // load first cpu buffer to device
    CHECK(cudaMemcpyAsync(io_buffers_gpu.at(input_idx), input_buffers_queue.front().get(), input_n_bytes,
                          cudaMemcpyHostToDevice, *h2d_stream_ptr));
    cudaStreamSynchronize(*h2d_stream_ptr);
    std::vector<uint64_t> time_diff;
    input_buffers_queue.pop();
    bool is_first_infer = true;
    while (not input_buffers_queue.empty()) {
        // infer with first device buffer
        auto start = NowUs();
        auto status = context->enqueueV2(io_buffers_gpu.data() + io_buffer_pos * num_bd, *infer_stream_ptr, nullptr);
        if (not status) {
            cout << "Enqueue resnet-18 async failed" << endl;
        }
        int next_i_buffer_pos = io_buffer_pos ^ 1;
        // load next cpu buffer to device
        CHECK(cudaMemcpyAsync(io_buffers_gpu.at(next_i_buffer_pos * num_bd + input_idx),
                              input_buffers_queue.front().get(), input_n_bytes, cudaMemcpyHostToDevice,
                              *h2d_stream_ptr));

        if (not is_first_infer) {
            std::shared_ptr<float> output_buffer;
            float* tmp_output_ptr;
            CHECK(cudaMallocHost(&tmp_output_ptr, output_n_volume * sizeof(float)));
            output_buffer.reset(tmp_output_ptr, [](float* p) { CHECK(cudaFreeHost(p)); });

            int last_o_buffer_pos = io_buffer_pos ^ 1;
            CHECK(cudaMemcpyAsync(output_buffer.get(), io_buffers_gpu.at(last_o_buffer_pos * num_bd + output_idx),
                                  output_n_bytes, cudaMemcpyDeviceToHost, *d2h_stream_ptr));
            cudaStreamSynchronize(*d2h_stream_ptr);
            // GetClassificationResult(output_buffer.get(), 1000, 5, 0);
            output_buffers_queue.push(output_buffer);
        }
        cudaStreamSynchronize(*h2d_stream_ptr);
        cudaStreamSynchronize(*infer_stream_ptr);

        input_buffers_queue.pop();
        io_buffer_pos ^= 1;
        is_first_infer = false;
        time_diff.push_back(NowUs() - start);
        cout << "After stream sync gap: " << time_diff.back() << endl;
    }
    auto status = context->enqueueV2(io_buffers_gpu.data() + io_buffer_pos * num_bd, *infer_stream_ptr, nullptr);
    if (not status) {
        cout << "Enqueue resnet-18 async failed" << endl;
    }
    auto last_output_buffer = std::shared_ptr<float>(new float[output_n_volume], ArrayDeleter());
    int last_o_buffer_pos = io_buffer_pos ^ 1;
    CHECK(cudaMemcpyAsync(last_output_buffer.get(), io_buffers_gpu.at(last_o_buffer_pos * num_bd + output_idx),
                          output_n_bytes, cudaMemcpyDeviceToHost, *d2h_stream_ptr));
    cudaStreamSynchronize(*d2h_stream_ptr);
    cudaStreamSynchronize(*infer_stream_ptr);
    output_buffers_queue.push(last_output_buffer);
    GetClassificationResult(output_buffers_queue.back().get(), 1000, 5, 0);

    auto output_buffer = std::shared_ptr<float>(new float[output_n_volume], ArrayDeleter());
    int o_buffer_pos = io_buffer_pos;
    CHECK(cudaMemcpyAsync(output_buffer.get(), io_buffers_gpu.at(o_buffer_pos * num_bd + output_idx), output_n_bytes,
                          cudaMemcpyDeviceToHost, *d2h_stream_ptr));
    cudaStreamSynchronize(*d2h_stream_ptr);
    output_buffers_queue.push(output_buffer);
    GetClassificationResult(output_buffers_queue.back().get(), 1000, 5, 0);

    // free device buffer
    for (void* ptr : io_buffers_gpu) cudaFree(ptr);
    for (void* ptr : warmup_io_buffers_gpu) cudaFree(ptr);
    cout << "Resnet-18 with IxRT API demo for enqueue hide data transfer done" << endl;
    int32_t batch_size = input_dims[0];
    float sum{0};
    for (const auto& diff : time_diff) {
        sum += diff;
    }
    float fps = 1 / ((sum / time_diff.size()) / 1000000) * batch_size;
    std::cout << "BatchSize: " << batch_size << " FPS: " << fps << std::endl;
}
