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

void WriteBuffer2Disk(const std::string& file_path, void* data, uint64_t len) {
    std::ofstream outFile(file_path, std::ios::binary);
    outFile.write((char*)data, len);
    outFile.close();
}

void InferenceYoloV5(const string& precision, const string& onnx_path, const string& quant_path,
                     const string& demo_image_path, const string& engine_path) {
    std::string input_name("images");
    std::vector<string> output_names = {"output"};
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (not builder) {
        std::cout << "Create builder failed" << std::endl;
        return;
    } else {
        cout << "Create builder success" << endl;
    }

    if (precision == "int8") {
        if (builder->platformHasFastInt8()) {
            cout << "Current support Int8 inference" << endl;
        } else {
            cout << "Current not support Int8 inference" << endl;
        }
    } else {
        if (builder->platformHasFastFp16()) {
            cout << "Current support Fp16 inference" << endl;
        } else {
            cout << "Current not support Fp16 inference" << endl;
        }
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

    if (precision == "int8")
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    else
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

    nvinfer1::Dims outputDims = network->getOutput(0)->getDimensions();
    ASSERT(outputDims.nbDims == 3);
    cout << "\nOutput dims: " << endl;
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << inputDims.d[i] << " ";
    }
    cout << endl;

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
        std::vector<char>* raw_plan = &buffer;
        input.read(raw_plan->data(), size);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(raw_plan->data(), plan->size()),
                                                        ObjectDeleter());
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
    for (const std::string& o_name : output_names) {
        auto output_idx = engine->getBindingIndex(o_name.c_str());
        cout << "Output index: " << output_idx << endl;
    }

    // allocate buffer
    std::vector<shared_ptr<float>> io_buffers_cpu;
    std::vector<void*> io_buffers_gpu(engine->getNbBindings());
    for (int i = 0; i < 2; ++i) {
        int n_volume = volume(engine->getBindingDimensions(i));
        int n_bytes = GetBytes(engine->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i), n_bytes));
        io_buffers_cpu.emplace_back(shared_ptr<float>(new float[n_volume], ArrayDeleter()));
        if (i == 0) {
            auto dims = Dims2Vec(engine->getBindingDimensions(i));
            LoadImageCPU(demo_image_path, io_buffers_cpu.at(0).get(), dims, 0, false);
            CHECK(cudaMemcpy(io_buffers_gpu.at(0), io_buffers_cpu.at(0).get(), n_bytes, cudaMemcpyHostToDevice));
        } else {
        }
    }
    cout << "User input date prepare done" << endl;

    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    puts("context->executeV2");
    // Inference
    auto status = context->executeV2(io_buffers_gpu.data());
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }
    puts("context->executeV2 Done");

    // Get Output (bsz, 8400*3, 6)
    int n_bytes = GetBytes(engine->getBindingDimensions(1), engine->getBindingDataType(1));
    CHECK(cudaMemcpy(io_buffers_cpu.at(1).get(), io_buffers_gpu.at(1), n_bytes, cudaMemcpyDeviceToHost));

    // Show result
    std::vector<DetectionResult> output_show;
    auto out_dims = engine->getBindingDimensions(1);
    YoloGetResults(io_buffers_cpu.at(1).get(), &output_show, 80, 3, 8400, 1, 3 * out_dims.d[2], 640, 640, 0);
    PrintDetectionResult(output_show);  // Free memory
    for (auto i : io_buffers_gpu) {
        CHECK(cudaFree(i));
    }
    cout << "YoloV5 Inference API demo done" << endl;
}
}  // namespace nvinfer1::samples

int main(int argc, char* argv[]) {
    std::string precision, datadir;
    std::string engine_path, onnx_path, quant_param_path = "";

    nvinfer1::samples::common::Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    initLibNvInferPlugins(&logger, "");

    CLI::App app{"App description"};
    app.add_option("--precision", precision);
    app.add_option("--data_dir", datadir);
    CLI11_PARSE(app, argc, argv);

    if (precision == "fp16") {
        engine_path = datadir + "/yolov5s_fp16.engine";
        onnx_path = datadir + "/yolov5_final.onnx";
        std::string quant_param_path = "";
    } else if (precision == "int8") {
        engine_path = datadir + "/yolov5s_int8_fusion.engine";
        onnx_path = datadir + "/yolov5_final.onnx";
        quant_param_path = datadir + "/quantized_yolov5s.json";
    } else {
        printf("Set Unsupported precision [%s], use [int8/fp16]\n", precision.c_str());
        return 0;
    }
    std::string image_path = datadir + "/dog_640.jpg";
    nvinfer1::samples::InferenceYoloV5(precision, onnx_path, quant_param_path, image_path, engine_path);
    return 0;
}
