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

#include "classification.h"

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include "NvInferImpl.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "error_recorder.h"
#include "image_io.h"
#include "json.hpp"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "postprocess_utils.h"

using json = nlohmann::json;
using std::cerr;
using std::cout;
using std::endl;

namespace nvinfer1::samples {
using namespace nvinfer1::samples::common;

static std::string dir_path("data/resnet18/");
void DumpBuffer2Disk(const std::string& file_path, void* data, uint64_t len) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (not out_file.is_open()) {
        out_file.close();
        std::exit(EXIT_FAILURE);
    }
    out_file.write((char*)data, len);
    out_file.close();
    cout << "Dump buffer size " << len << endl;
}
void LoadBufferFromDisk(const std::string& file_path, std::vector<int8_t>* engine_buffer) {
    std::ifstream in_file(file_path, std::ios::binary);
    if (not in_file.is_open()) {
        in_file.close();
        std::exit(EXIT_FAILURE);
    }
    in_file.seekg(0, std::ios::end);
    uint64_t file_length = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    engine_buffer->resize(file_length);
    in_file.read((char*)engine_buffer->data(), file_length);
    in_file.close();
    cout << "Load buffer size " << file_length << endl;
}

void PrintDims(const nvinfer1::Dims& dim, const std::string& prefix = "") {
    cout << prefix << endl;
    for (auto i = 0; i < dim.nbDims; ++i) {
        cout << dim.d[i] << " ";
    }
    cout << endl;
}

void MyHook(nvinfer1::ExecutionContextInfo const* info) {
    std::cout << "Running callback function " << info->hookName << std::endl;
    std::cout << "Running op " << info->opName << std::endl;
    std::cout << "NbInputs " << info->nbInputs << std::endl;
    std::cout << "NbOutputs " << info->nbOutputs << std::endl;

    for (int32_t i = 0; i < info->nbInputs; i++) {
        std::cout << i << "th input is " << info->inputNames[i] << std::endl;
    }
}

void IxRTAPIExecute(const std::string& model_path, const std::string& quant_file, const std::string& engine_save_path,
                    nvinfer1::BuilderFlag flag) {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(flag);

    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk(engine_save_path, &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTAPIExecuteImplicitQuantization(const std::string& model_path, const std::string& quant_file,
                                        const std::string& engine_save_path, nvinfer1::BuilderFlag flag) {
    std::ifstream in(quant_file);
    json quantization_config;
    in >> quantization_config;
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(flag);

    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    // setDynamicRange
    int nb_layers = network->getNbLayers();
    std::cout << "nb_layers: " << nb_layers << std::endl;
    for (int i = 0; i < nb_layers; i++) {
        auto layer = network->getLayer(i);
        int nb_layer_inputs = layer->getNbInputs();
        int nb_layer_outputs = layer->getNbOutputs();
        for (int j = 0; j < nb_layer_inputs; j++) {
            auto tensor = layer->getInput(j);
            if (tensor and tensor->isNetworkInput()) {
                auto tensor_name = tensor->getName();
                if (quantization_config.contains(tensor_name)) {
                    auto tensor_q_config = quantization_config[tensor_name];
                    float min_v = tensor_q_config["min"];
                    float max_v = tensor_q_config["max"];
                    tensor->setDynamicRange(min_v, max_v);
                    // std::cout << tensor_name << " set min: " << min_v << ", max: " << max_v << std::endl;
                }
            }
        }
        for (int j = 0; j < nb_layer_outputs; j++) {
            auto tensor = layer->getOutput(j);
            auto tensor_name = tensor->getName();
            if (quantization_config.contains(tensor_name)) {
                auto tensor_q_config = quantization_config[tensor_name];
                float min_v = tensor_q_config["min"];
                float max_v = tensor_q_config["max"];
                tensor->setDynamicRange(min_v, max_v);
                // std::cout << tensor_name << " set min: " << min_v << ", max: " << max_v << std::endl;
            }
        }
    }

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk(engine_save_path, &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 implicit quantization with IxRT API demo done" << endl;
}

void IxRTAPIExecuteFromSerializedONNX() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string model_path(dir_path + "resnet18_qdq_external.onnx");
    std::string external_weight_location(dir_path + "resnet18_qdq_external_data");
    std::string input_name("input");
    std::string output_name("output");
    std::string engine_save_path(dir_path + "resnet18_qdq.engine");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }

    std::ifstream onnx_file(model_path, std::ios::ate | std::ios::binary);
    std::streamsize onnx_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buffer(onnx_size);
    if (!onnx_file.read(onnx_buffer.data(), onnx_size)) {
        std::cout << "Failed to read from file: " << model_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    auto parsed = parser->parse(onnx_buffer.data(), onnx_size, external_weight_location.c_str());

    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk(engine_save_path, &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTAPIEnqueue(bool use_enqueue_v3) {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string model_path(dir_path + "resnet18_qdq.onnx");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kWARNING);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    // Option operation, not necessary
    // WriteBuffer2Disk("/home/work/trt.engine", plan->data(), plan->size());

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    std::vector<void*> warm_up_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    CHECK(cudaMalloc(&warm_up_buffer.at(0), input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMalloc(&warm_up_buffer.at(1), output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    auto stream_ptr = makeCudaStream();
    auto event_ptr = makeCudaEvent();
    for (auto i = 0; i < 10; ++i) {
        auto status = context->enqueueV2(warm_up_buffer.data(), *stream_ptr, nullptr);
        if (status) {
            cout << "Warm up enqueue resnet-18 async success" << endl;
        } else {
            cout << "Warm up enqueue resnet-18 async failed" << endl;
        }
    }
    cudaStreamSynchronize(*stream_ptr);
    std::cout << "Warm up done" << endl;
    auto start = NowUs();
    bool status{false};
    if (use_enqueue_v3) {
        context->setInputConsumedEvent(*event_ptr);
        if (context->setTensorAddress(input_name.c_str(), binding_buffer.at(0))) {
            cout << "Set input data done" << endl;
        }
        if (context->setTensorAddress(output_name.c_str(), binding_buffer.at(1))) {
            cout << "Set output data done" << endl;
        }
        cout << "Enqueue V3" << endl;
        status = context->enqueueV3(*stream_ptr);
    } else {
        cout << "Enqueue V3" << endl;
        status = context->enqueueV2(binding_buffer.data(), *stream_ptr, event_ptr.get());
    }
    cout << "After enqueue gap: " << NowUs() - start << endl;
    if (status) {
        cout << "Enqueue resnet-18 async success" << endl;
    } else {
        cout << "Enqueue resnet-18 async failed" << endl;
    }
    cudaEventSynchronize(*event_ptr);
    cout << "After event synchronized, gap: " << NowUs() - start << endl;
    cudaStreamSynchronize(*stream_ptr);
    cout << "After stream sync gap: " << NowUs() - start << endl;
    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo for enqueue done" << endl;
}

void IxRTAPIMultiContext() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string image_path_2(dir_path + "robin_224.bmp");
    std::string model_path(dir_path + "resnet18.onnx");
    std::string quant_file(dir_path + "quantized_resnet18_shape_opset11.json");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create onnx parser success" << endl;
    }

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

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    auto cpu_fp32_image_2 = LoadImageCPU(image_path_2, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    std::vector<void*> binding_buffer_2(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    std::shared_ptr<float> cpu_output_2(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    void* input_gpu_2{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    CHECK(cudaMalloc(&input_gpu_2, input_size));
    void* output_gpu{nullptr};
    void* output_gpu_2{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMalloc(&output_gpu_2, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(input_gpu_2, cpu_fp32_image_2.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    binding_buffer_2.at(0) = input_gpu_2;
    binding_buffer_2.at(1) = output_gpu_2;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto context_2 = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    auto status_2 = context_2->executeV2(binding_buffer_2.data());
    if (not status_2) {
        cerr << "Context 2 execute ixrt failed" << endl;
    } else {
        cout << "Context 2 execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_output_2.get(), output_gpu_2, output_size, cudaMemcpyDeviceToHost));
    cout << "Result 1: " << endl;
    GetClassificationResult(cpu_output.get(), 1000, 5);
    cout << "Result 2: " << endl;
    GetClassificationResult(cpu_output_2.get(), 1000, 5);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    CHECK(cudaFree(input_gpu_2));
    CHECK(cudaFree(output_gpu_2));
}

void IxRTAPIDynamicShape() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string image_path_2(dir_path + "kitten_196.bmp");
    std::string model_path(dir_path + "resnet18-all-dynamic.onnx");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{4, {1, 3, 112, 112}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{4, {1, 3, 224, 224}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{4, {1, 3, 448, 448}});
    config->addOptimizationProfile(profile);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create onnx parser success" << endl;
    }

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

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    nvinfer1::Dims dynamic_input_dims{4, {1, 3, 224, 224}};
    auto cpu_fp32_image = LoadImageCPU(image_path, dynamic_input_dims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors() * engine->getNbOptimizationProfiles());
    auto input_size = GetBytes(dynamic_input_dims, engine->getTensorDataType(input_name.c_str()));
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));

    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;

    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    context->setInputShape(input_name.c_str(), dynamic_input_dims);
    auto context_input_dims = context->getTensorShape(input_name.c_str());
    cout << "Dynamic input dims: ";
    for (auto i = 0; i < context_input_dims.nbDims; ++i) {
        cout << context_input_dims.d[i] << " ";
    }
    cout << endl;
    auto context_output_dims = context->getTensorShape(output_name.c_str());
    cout << "Dynamic output dims: ";
    for (auto i = 0; i < context_output_dims.nbDims; ++i) {
        cout << context_output_dims.d[i] << " ";
    }
    cout << endl;

    auto output_size = GetBytes(context_output_dims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    std::shared_ptr<float> cpu_output_2(new float[output_size / sizeof(float)], ArrayDeleter());
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));

    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    cout << "Big image result" << endl;
    GetClassificationResult(cpu_output.get(), 1000, 5);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));

    // Prepare image with small size
    nvinfer1::Dims dynamic_input_dims_2{4, {1, 3, 196, 196}};
    auto cpu_fp32_image_2 = LoadImageCPU(image_path_2, dynamic_input_dims_2);
    auto input_size_2 = GetBytes(dynamic_input_dims_2, engine->getTensorDataType(input_name.c_str()));
    void* input_gpu_2{nullptr};
    CHECK(cudaMalloc(&input_gpu_2, input_size_2));
    CHECK(cudaMemcpy(input_gpu_2, cpu_fp32_image_2.get(), input_size_2, cudaMemcpyHostToDevice));

    context->setInputShape(input_name.c_str(), dynamic_input_dims_2);

    auto context_output_dims_2 = context->getTensorShape(output_name.c_str());

    auto output_size_2 = GetBytes(context_output_dims_2, engine->getTensorDataType(output_name.c_str()));
    void* output_gpu_2{nullptr};
    CHECK(cudaMalloc(&output_gpu_2, output_size_2));
    std::vector<void*> binding_buffer_2(engine->getNbIOTensors());
    binding_buffer_2.at(0) = input_gpu_2;
    binding_buffer_2.at(1) = output_gpu_2;
    auto status_2 = context->executeV2(binding_buffer_2.data());
    if (not status_2) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    CHECK(cudaMemcpy(cpu_output_2.get(), output_gpu_2, output_size_2, cudaMemcpyDeviceToHost));
    cout << "Small image result" << endl;
    GetClassificationResult(cpu_output_2.get(), 1000, 5);
    CHECK(cudaFree(input_gpu_2));
    CHECK(cudaFree(output_gpu_2));
    cout << "Done" << endl;
}

void IxRTAPIDynamicShapeMultiContext() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string image_path_2(dir_path + "robin_224.bmp");
    std::string model_path(dir_path + "resnet18-all-dynamic.onnx");
    std::string input_name("input");
    std::string output_name("output");

    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{4, {1, 3, 112, 112}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{4, {1, 3, 224, 224}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{4, {1, 3, 448, 448}});
    config->addOptimizationProfile(profile);
    // Add second profile
    auto profile_2 = builder->createOptimizationProfile();
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                             nvinfer1::Dims{4, {1, 3, 112, 112}});
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                             nvinfer1::Dims{4, {1, 3, 224, 224}});
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                             nvinfer1::Dims{4, {1, 3, 448, 448}});
    config->addOptimizationProfile(profile_2);
    // Add third profile, same as first
    config->addOptimizationProfile(profile);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create onnx parser success" << endl;
    }

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

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    int32_t profile_index = 1;
    auto input_min_dim = engine->getProfileShape(input_name.c_str(), profile_index, nvinfer1::OptProfileSelector::kMIN);
    PrintDims(input_min_dim, "Input min dim: ");
    auto input_opt_dim = engine->getProfileShape(input_name.c_str(), profile_index, nvinfer1::OptProfileSelector::kOPT);
    PrintDims(input_opt_dim, "Input opt dim: ");
    auto input_max_dim = engine->getProfileShape(input_name.c_str(), profile_index, nvinfer1::OptProfileSelector::kMAX);
    PrintDims(input_max_dim, "Input max dim: ");

    auto num_io_tensor = engine->getNbIOTensors();
    cout << "Number of io " << num_io_tensor << endl;
    auto num_profile = engine->getNbOptimizationProfiles();
    cout << "Number of profile " << num_profile << endl;
    std::vector<void*> binding_buffer(num_io_tensor * num_profile);

    nvinfer1::Dims dynamic_input_dims{4, {1, 3, 224, 224}};
    auto cpu_fp32_image = LoadImageCPU(image_path, dynamic_input_dims);
    auto input_size = GetBytes(dynamic_input_dims, engine->getTensorDataType(input_name.c_str()));
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));

    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;

    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    context->setInputShape(input_name.c_str(), dynamic_input_dims);
    auto context_input_dims = context->getTensorShape(input_name.c_str());
    cout << "Dynamic input dims: ";
    for (auto i = 0; i < context_input_dims.nbDims; ++i) {
        cout << context_input_dims.d[i] << " ";
    }
    cout << endl;
    auto context_output_dims = context->getTensorShape(output_name.c_str());
    cout << "Dynamic output dims: ";
    for (auto i = 0; i < context_output_dims.nbDims; ++i) {
        cout << context_output_dims.d[i] << " ";
    }
    cout << endl;

    auto output_size = GetBytes(context_output_dims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));

    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    cout << "First context with profile 0 image result" << endl;
    GetClassificationResult(cpu_output.get(), 1000, 5);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));

    // Prepare image with small size
    auto context_2 = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    auto stream = makeCudaStream();
    context_2->setOptimizationProfileAsync(profile_index, *stream);

    auto binding_cell_offset = profile_index * num_io_tensor;
    cout << "Binding cell offset: " << binding_cell_offset << endl;

    // Switch optimization profile by set profile index, context resource will be updated
    //    context_2->setOptimizationProfileAsync(2, *stream);

    nvinfer1::Dims dynamic_input_dims_2{4, {1, 3, 224, 224}};
    auto cpu_fp32_image_2 = LoadImageCPU(image_path_2, dynamic_input_dims_2);
    auto input_size_2 = GetBytes(dynamic_input_dims_2, engine->getTensorDataType(input_name.c_str()));
    void* input_gpu_2{nullptr};
    CHECK(cudaMalloc(&input_gpu_2, input_size_2));
    CHECK(cudaMemcpy(input_gpu_2, cpu_fp32_image_2.get(), input_size_2, cudaMemcpyHostToDevice));

    context_2->setInputShape(input_name.c_str(), dynamic_input_dims_2);
    auto context_output_dims_2 = context_2->getTensorShape(output_name.c_str());

    auto output_size_2 = GetBytes(context_output_dims_2, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output_2(new float[output_size_2 / sizeof(float)], ArrayDeleter());
    void* output_gpu_2{nullptr};
    CHECK(cudaMalloc(&output_gpu_2, output_size_2));
    binding_buffer.at(0 + binding_cell_offset) = input_gpu_2;
    binding_buffer.at(1 + binding_cell_offset) = output_gpu_2;
    auto status_2 = context_2->executeV2(binding_buffer.data());
    if (not status_2) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    CHECK(cudaMemcpy(cpu_output_2.get(), output_gpu_2, output_size_2, cudaMemcpyDeviceToHost));
    cout << "Second context with profile 1 image result" << endl;
    GetClassificationResult(cpu_output_2.get(), 1000, 5);
    CHECK(cudaFree(input_gpu_2));
    CHECK(cudaFree(output_gpu_2));
    cout << "Done" << endl;
}

void IxRTAPILoadEngine() {
    std::string engine_save_path("/tmp/resnet18_i8.engine");
    std::string image_path("data/resnet18/kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    cout << "Load engine from" << engine_save_path << endl;

    SampleErrorRecorder my_error_recorder;
    runtime->setErrorRecorder(&my_error_recorder);
    std::vector<int8_t> engine_buffer;
    LoadBufferFromDisk(engine_save_path, &engine_buffer);
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    if (auto* er = runtime->getErrorRecorder()) {
        if (er->getNbErrors()) {
            std::cerr << "Got " << er->getNbErrors() << " errors from IxRT:" << std::endl;
            for (int i = 0; i < er->getNbErrors(); ++i) {
                std::cerr << "Error #" << i << std::endl;
                std::cerr << "Error code:" << int32_t(er->getErrorCode(i)) << std::endl;
                std::cerr << "Error desc:" << er->getErrorDesc(i) << std::endl;
            }
            throw "Failed to load engine";
        }
    } else {
        throw "Fatal: got nullptr error recorder from engine";
    }
    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    auto inputDims = engine->getTensorShape(input_name.c_str());
    auto outputDims = engine->getTensorShape(output_name.c_str());
    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    auto again = UPtr<nvinfer1::IHostMemory>(context->getEngine().serialize());
    if (again) {
        cout << "Create again success size: " << again->size() << endl;
    } else {
        cout << "Create again failed" << endl;
    }
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTAPIExecuteWithHook(const std::string& model_path, const std::string& quant_file,
                            const std::string& engine_save_path) {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk("/tmp/resnet18_trt.engine", &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
#ifndef USE_TRT
    context->registerHook("Print name", MyHook, nvinfer1::ExecutionHookFlag::kPRERUN);
#endif
    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTAPIExecuteCustomFP32Layers(const std::string& model_path, const std::string& engine_save_path,
                                    nvinfer1::BuilderFlag flag) {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create builder success" << endl;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (not network) {
        std::cout << "Create network failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(flag);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < network->getNbLayers(); i++) {
        nvinfer1::ILayer* layer = network->getLayer(i);
        std::cout << layer->getName() << std::endl;
        if (layer->getType() == nvinfer1::LayerType::kCONVOLUTION) {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
        }
        if (layer->getType() == nvinfer1::LayerType::kACTIVATION) {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
        }
        if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
        }
    }
    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk("/tmp/resnet18_trt.engine", &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTContextMemoryExecute(const std::string& model_path, const std::string& quant_file,
                              const std::string& engine_save_path) {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
        LoadBufferFromDisk(engine_save_path, &engine_buffer);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()), ObjectDeleter());
    } else {
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                        ObjectDeleter());
    }

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());
    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbIOTensors());
    auto input_size = GetBytes(inputDims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    size_t context_mem_size = engine->getDeviceMemorySize();
    void* context_gpu{nullptr};
    CHECK(cudaMalloc(&context_gpu, context_mem_size));
    context->setDeviceMemory(context_gpu);

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Resnet-18 with IxRT API demo done" << endl;
}

void IxRTContextMemoryExecuteDynamic() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string image_path_2(dir_path + "robin_224.bmp");
    std::string model_path(dir_path + "resnet18-all-dynamic.onnx");
    std::string input_name("input");
    std::string output_name("output");
    Logger logger(nvinfer1::ILogger::Severity::kINFO);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        std::exit(EXIT_FAILURE);
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
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create network success" << endl;
    }

    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{4, {1, 3, 112, 112}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{4, {1, 3, 224, 224}});
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{4, {1, 3, 448, 448}});
    config->addOptimizationProfile(profile);
    // Add second profile
    auto profile_2 = builder->createOptimizationProfile();
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                             nvinfer1::Dims{4, {1, 3, 112, 112}});
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                             nvinfer1::Dims{4, {1, 3, 196, 196}});
    profile_2->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                             nvinfer1::Dims{4, {1, 3, 224, 224}});
    config->addOptimizationProfile(profile_2);

    auto parser = UPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create config success" << endl;
    }
    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        std::exit(EXIT_FAILURE);
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

    auto input_type = network->getInput(0)->getType();
    cout << "Input dtype:" << (int32_t)input_type << endl;
    auto output_type = network->getInput(0)->getType();
    cout << "Output dtype:" << (int32_t)output_type << endl;

    cout << "\nOutput dimes: " << endl;
    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 2);
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    UPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        cout << "Create runtime done" << endl;
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<int8_t> engine_buffer;
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                    ObjectDeleter());

    if (not engine) {
        std::cout << "Create engine failed" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Create engine done" << endl;
    }

    cout << "Engine name: " << engine->getName() << endl;
    auto num_bd = engine->getNbIOTensors();
    cout << "Number of binding data: " << num_bd << endl;
    for (auto i = 0; i < num_bd; ++i) {
        cout << "The " << i << " binding" << endl;

        auto tensor_name = engine->getIOTensorName(i);
        cout << "Name: " << tensor_name << endl;
        cout << "Format: " << (int32_t)engine->getTensorFormat(tensor_name) << endl;
        cout << "Data type: " << (int32_t)engine->getTensorDataType(tensor_name) << endl;

        cout << "Dimension: ";
        auto dim = engine->getTensorShape(tensor_name);
        for (auto k = 0; k < dim.nbDims; ++k) {
            cout << dim.d[k] << " ";
        }
        cout << endl;
    }

    inputDims = engine->getTensorShape(input_name.c_str());
    outputDims = engine->getTensorShape(output_name.c_str());

    nvinfer1::Dims dynamic_input_dims{4, {1, 3, 224, 224}};
    auto cpu_fp32_image = LoadImageCPU(image_path, dynamic_input_dims);

    auto num_io_tensor = engine->getNbIOTensors();
    cout << "Number of io " << num_io_tensor << endl;
    auto num_profile = engine->getNbOptimizationProfiles();
    cout << "Number of profile " << num_profile << endl;

    std::vector<void*> binding_buffer(num_io_tensor * num_profile);
    auto input_size = GetBytes(dynamic_input_dims, engine->getTensorDataType(input_name.c_str()));
    auto output_size = GetBytes(outputDims, engine->getTensorDataType(output_name.c_str()));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    std::shared_ptr<float> cpu_output_2(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(0) = input_gpu;
    binding_buffer.at(1) = output_gpu;
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    size_t context_mem_size = engine->getDeviceMemorySize();
    std::cout << "### context_mem_size: " << context_mem_size << std::endl;
    void* context_gpu{nullptr};
    CHECK(cudaMalloc(&context_gpu, context_mem_size));
    context->setDeviceMemory(context_gpu);

    context->setInputShape(input_name.c_str(), dynamic_input_dims);

    auto status = context->executeV2(binding_buffer.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);
    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));

    // Prepare image with small size
    auto context_2 = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    auto stream = makeCudaStream();
    int32_t profile_index = 1;
    context_2->setOptimizationProfileAsync(profile_index, *stream);
    // Switch optimization profile by set profile index, context resource will be updated
    //    context_2->setOptimizationProfileAsync(2, *stream);

    auto binding_cell_offset = profile_index * num_io_tensor;
    cout << "Binding cell offset: " << binding_cell_offset << endl;

    nvinfer1::Dims dynamic_input_dims_2{4, {1, 3, 224, 224}};
    auto cpu_fp32_image_2 = LoadImageCPU(image_path_2, dynamic_input_dims_2);
    auto input_size_2 = GetBytes(dynamic_input_dims_2, engine->getTensorDataType(input_name.c_str()));
    void* input_gpu_2{nullptr};
    CHECK(cudaMalloc(&input_gpu_2, input_size_2));
    CHECK(cudaMemcpy(input_gpu_2, cpu_fp32_image_2.get(), input_size_2, cudaMemcpyHostToDevice));

    context_2->setInputShape(input_name.c_str(), dynamic_input_dims_2);

    auto context_output_dims_2 = context_2->getTensorShape(output_name.c_str());

    auto output_size_2 = GetBytes(context_output_dims_2, engine->getTensorDataType(output_name.c_str()));
    void* output_gpu_2{nullptr};
    CHECK(cudaMalloc(&output_gpu_2, output_size_2));
    binding_buffer.at(0 + binding_cell_offset) = input_gpu_2;
    binding_buffer.at(1 + binding_cell_offset) = output_gpu_2;
    auto status_2 = context_2->executeV2(binding_buffer.data());
    if (not status_2) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    CHECK(cudaMemcpy(cpu_output_2.get(), output_gpu_2, output_size_2, cudaMemcpyDeviceToHost));
    cout << "Second context with profile 1 image result" << endl;
    GetClassificationResult(cpu_output_2.get(), 1000, 5);
    CHECK(cudaFree(input_gpu_2));
    CHECK(cudaFree(output_gpu_2));

    cout << "Resnet-18 with ContextMemoryExecuteDynamic demo done" << endl;
}

}  // namespace nvinfer1::samples
