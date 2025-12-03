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

#include <dlfcn.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "postprocess_utils.h"

using namespace std;
namespace nvinfer1::samples {
using namespace common;

void InferenceYoloV3DeprecatedAPI(const string &onnx_path, const string &quant_path, const string &demo_image_path,
                                  const string &engine_path);

void WriteBuffer2Disk(const std::string &file_path, void *data, uint64_t len) {
    std::ofstream outFile(file_path, std::ios::binary);
    outFile.write((char *)data, len);
    outFile.close();
}

void PrintDims(const nvinfer1::Dims &dim, const std::string &prefix = "") {
    cout << prefix << endl;
    for (auto i = 0; i < dim.nbDims; ++i) {
        cout << dim.d[i] << " ";
    }
    cout << endl;
}

/* Execute YoloV3 with input having width of `width` */
void ExecuteYoloV3(int width, nvinfer1::IExecutionContext *context, const vector<int64_t> &io_indexes,
                   nvinfer1::ICudaEngine *engine) {
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "Execute YoloV3 with input of (1,3," << width << "," << width << ")" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    // Step1: context->setBindingDimensions
    nvinfer1::Dims input_dims{4, {1, 3, width, width}};
    context->setBindingDimensions(io_indexes.at(0), input_dims);
    // Step2: Allocate IO buffer
    std::vector<shared_ptr<float>> io_buffers_cpu;
    std::vector<void *> io_buffers_gpu(engine->getNbBindings());
    for (int i = 0; i < 4; ++i) {
        int n_volume = volume(context->getBindingDimensions(i));
        int n_bytes = GetBytes(context->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i), n_bytes));
        io_buffers_cpu.emplace_back(shared_ptr<float>(new float[n_volume], ArrayDeleter()));
        if (i == 0) {
            auto dims = Dims2Vec(context->getBindingDimensions(i));
            LoadImageCPU("data/yolov3/dog_" + std::to_string(width) + ".bmp", io_buffers_cpu.at(0).get(), dims, 0,
                         false);
            CHECK(cudaMemcpy(io_buffers_gpu.at(0), io_buffers_cpu.at(0).get(), n_bytes, cudaMemcpyHostToDevice));
        }
    }
    cout << "User input data prepare done" << endl;
    // Step3: Inference
    bool status = context->executeV2(io_buffers_gpu.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    // Step4: Show result
    std::vector<DetectionResult> output_show;
    for (int i = 1; i <= 3; ++i) {
        int n_bytes = GetBytes(context->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMemcpy(io_buffers_cpu.at(i).get(), io_buffers_gpu.at(i), n_bytes, cudaMemcpyDeviceToHost));
        int feat_len = context->getBindingDimensions(i).d[1], box_info_len = context->getBindingDimensions(i).d[2];
        YoloGetResults(io_buffers_cpu.at(i).get(), &output_show, 80, 3, feat_len / 3, 1, 3 * box_info_len, width, width,
                       0);
    }
    PrintDetectionResult(output_show);
    // Step5: Free memory
    for (auto i : io_buffers_gpu) {
        CHECK(cudaFree(i));
    }
}

void InferenceYoloV3(const string &onnx_path, const string &demo_image_path, const string &engine_path) {
    std::string input_name("images");
    std::vector<string> output_names = {"decoder_52", "decoder_26", "decoder_13"};
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
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
    auto network = UPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (not network) {
        std::cout << "Create network failed" << std::endl;
        return;
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
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
    cout << "\nInput dims: " << endl;
    for (auto i = 0; i < inputDims.nbDims; ++i) {
        cout << inputDims.d[i] << " ";
    }

    cout << "\nOutput dims: " << endl;
    for (int i = 0; i < 3; ++i) {
        auto oup_dims = network->getOutput(i)->getDimensions();
        ASSERT(oup_dims.nbDims == 3);
        for (int j = 0; j < oup_dims.nbDims; ++j) {
            cout << oup_dims.d[j] << ",";
        }
        cout << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        return;
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<char> buffer;
    UPtr<nvinfer1::IHostMemory> plan;
    if (false) {
        std::ifstream input(engine_path, std::ios::ate | std::ios::binary);
        // get current position in file
        std::streamsize size = input.tellg();
        // move to start of file
        input.seekg(0, std::ios::beg);
        // read raw data
        buffer.resize(size);
        std::vector<char> *raw_plan = &buffer;
        input.read(raw_plan->data(), size);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(raw_plan->data(), raw_plan->size()), ObjectDeleter());
    } else {
        plan = UPtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (not plan) {
            std::cout << "Create serialized engine plan failed" << std::endl;
            return;
        } else {
            cout << "Create serialized engine plan done" << endl;
        }
        if (not engine_path.empty()) {
            WriteBuffer2Disk(engine_path, plan->data(), plan->size());
        }
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }
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
    for (const std::string &o_name : output_names) {
        auto output_idx = engine->getBindingIndex(o_name.c_str());
        cout << "Output index: " << output_idx << endl;
    }

    // allocate buffer
    std::vector<shared_ptr<float>> io_buffers_cpu;
    std::vector<void *> io_buffers_gpu(engine->getNbBindings());
    for (int i = 0; i < 4; ++i) {
        int n_volume = volume(engine->getBindingDimensions(i));
        int n_bytes = GetBytes(engine->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i), n_bytes));
        io_buffers_cpu.emplace_back(shared_ptr<float>(new float[n_volume], ArrayDeleter()));
        if (i == 0) {
            auto dims = Dims2Vec(engine->getBindingDimensions(i));
            LoadImageCPU(demo_image_path, io_buffers_cpu.at(0).get(), dims, 0, false);
            CHECK(cudaMemcpy(io_buffers_gpu.at(0), io_buffers_cpu.at(0).get(), n_bytes, cudaMemcpyHostToDevice));
        }
    }
    cout << "User input date prepare done" << endl;

    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    // Inference
    auto status = context->executeV2(io_buffers_gpu.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    // Show result
    std::vector<DetectionResult> output_show;
    for (int i = 1; i <= 3; ++i) {
        int n_bytes = GetBytes(engine->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMemcpy(io_buffers_cpu.at(i).get(), io_buffers_gpu.at(i), n_bytes, cudaMemcpyDeviceToHost));
        int feat_len = engine->getBindingDimensions(i).d[1], box_info_len = engine->getBindingDimensions(i).d[2];
        YoloGetResults(io_buffers_cpu.at(i).get(), &output_show, 80, 3, feat_len / 3, 1, 3 * box_info_len, 416, 416, 0);
    }
    PrintDetectionResult(output_show);  // Free memory
    for (auto i : io_buffers_gpu) {
        CHECK(cudaFree(i));
    }
    cout << "YoloV3 Inference API demo done" << endl;
}

void InferenceYoloV3DynamicShape(const string &onnx_path, const string &demo_image_path, const string &engine_path) {
    std::string input_name("images");
    std::vector<string> output_names = {"decoder_52", "decoder_26", "decoder_13"};
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
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
    auto network = UPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (not network) {
        std::cout << "Create network failed" << std::endl;
        return;
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // Dynamic Shape: use size range
    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims max_dims{4, {1, 3, 608, 608}};
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{4, {1, 3, 288, 288}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{4, {1, 3, 320, 320}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    config->addOptimizationProfile(profile);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
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
    cout << "\nInput dims: " << endl;
    for (auto i = 0; i < inputDims.nbDims; ++i) {
        cout << inputDims.d[i] << " ";
    }

    cout << "\nOutput dims: " << endl;
    for (int i = 0; i < 3; ++i) {
        auto oup_dims = network->getOutput(i)->getDimensions();
        ASSERT(oup_dims.nbDims == 3);
        for (int j = 0; j < oup_dims.nbDims; ++j) {
            cout << oup_dims.d[j] << ",";
        }
        cout << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        return;
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<char> buffer;
    UPtr<nvinfer1::IHostMemory> plan;
    if (false) {
        std::ifstream input(engine_path, std::ios::ate | std::ios::binary);
        // get current position in file
        std::streamsize size = input.tellg();
        // move to start of file
        input.seekg(0, std::ios::beg);
        // read raw data
        buffer.resize(size);
        std::vector<char> *raw_plan = &buffer;
        input.read(raw_plan->data(), size);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(raw_plan->data(), raw_plan->size()), ObjectDeleter());
    } else {
        plan = UPtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (not plan) {
            std::cout << "Create serialized engine plan failed" << std::endl;
            return;
        } else {
            cout << "Create serialized engine plan done" << endl;
        }
        if (not engine_path.empty()) {
            WriteBuffer2Disk(engine_path, plan->data(), plan->size());
        }
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        return;
    } else {
        std::cout << "Create engine done" << endl;
    }

    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    vector<int64_t> io_indexes{engine->getBindingIndex(input_name.c_str())};
    for (auto o_name : output_names) {
        io_indexes.emplace_back(engine->getBindingIndex(o_name.c_str()));
    }
    ExecuteYoloV3(416, context.get(), io_indexes, engine.get());
    ExecuteYoloV3(608, context.get(), io_indexes, engine.get());
    ExecuteYoloV3(288, context.get(), io_indexes, engine.get());
    ExecuteYoloV3(320, context.get(), io_indexes, engine.get());
    cout << "YoloV3 Inference API demo done" << endl;
}
}  // namespace nvinfer1::samples
int main(int argc, char *argv[]) {
    nvinfer1::samples::common::Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    initLibNvInferPlugins(&logger, "");
    CLI::App app{"App description"};

    std::string onnx_path = "data/yolov3/quantized_yolov3_with_decoder.onnx";
    std::string image_path = "data/yolov3/dog_416.bmp";
    std::string demo = "trt_exe";
    string engine_path = "";
    app.add_option("--onnx", onnx_path, "YoloV3 onnx model path");
    app.add_option("-e, --engine", engine_path, "YoloV3 engine path, will build one if not exists");
    app.add_option("-i, --input", image_path, "Test image");
    app.add_option("--demo", demo, "Choose which demo to run, available: old/trt_exec/trt_dyn");
    CLI11_PARSE(app, argc, argv);
    if (demo == "trt_exe") {
        nvinfer1::samples::InferenceYoloV3(onnx_path, image_path, engine_path);
    } else if (demo == "trt_dyn") {
        nvinfer1::samples::InferenceYoloV3DynamicShape(onnx_path, image_path, engine_path);
    } else {
        throw "No such demo called " + demo + ", available: (old/trt_exe/trt_dyn)";
    }
    return 0;
}
