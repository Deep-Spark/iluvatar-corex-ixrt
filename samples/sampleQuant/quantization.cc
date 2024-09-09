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


#include "quantization.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "batch_stream.h"
#include "entropy_calibration.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"

using std::cerr;
using std::cout;
using std::endl;

void DumpBuffer2Disk(const std::string& file_path, void* data, uint64_t len) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (not out_file.is_open()) {
        out_file.close();
        return;
    }
    out_file.write((char*)data, len);
    out_file.close();
    cout << "Dump buffer size " << len << endl;
}

void LoadBufferFromDisk(const std::string& file_path, std::vector<int8_t>* engine_buffer) {
    std::ifstream in_file(file_path, std::ios::binary);
    if (not in_file.is_open()) {
        in_file.close();
        return;
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

void PrintNetworkInfo(nvinfer1::INetworkDefinition* network) {
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
}

void PrintEngineInfo(nvinfer1::ICudaEngine* engine, std::string input_name, std::string output_name) {
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
}

void IxRTAPIQuant(const std::string& model_path, const std::string& engine_save_path) {
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
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
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (not parser) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }

    bool parsed = false;
    parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(logger.getReportableSeverity()));
    if (!parsed) {
        std::cout << "Create onnx parser failed" << std::endl;
        return;
    } else {
        cout << "Create onnx parser success" << endl;
    }

    PrintNetworkInfo(network.get());

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    std::vector<std::string> dir_names{"data/IxRTQuant/n1111"};
    SampleHelper::BatchStream batch_stream(1, 8, 224, 224, 500, dir_names);
    calibrator.reset(
        new nvinfer1::Int8EntropyCalibrator2<SampleHelper::BatchStream>(batch_stream, 0, "_resnet50", {"input"}));
    config->setInt8Calibrator(calibrator.get());

    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        return;
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    if (not engine_save_path.empty()) {
        DumpBuffer2Disk(engine_save_path, plan->data(), plan->size());
    }
}
