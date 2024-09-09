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
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "CLI11.hpp"
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "coco_labels.h"
#include "data_utils.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"

using namespace std;
using namespace nvinfer1;

void WriteBuffer2Disk(const std::string& file_path, void* data, uint64_t len) {
    std::ofstream outFile(file_path, std::ios::binary);
    outFile.write((char*)data, len);
    outFile.close();
}

void PrintYoloXResult(float const* const count_ptr, float const* const nmsed_box_ptr,
                      float const* const nmsed_score_ptr, float const* const nmsed_class_ptr) {
    int32_t num_detections = (int32_t)count_ptr[0];

    std::cout << "Total detection output size: " << num_detections << std::endl;
    size_t box_offset = 0;
    for (int32_t i = 0; i < num_detections; i++) {
        printf("%dth lable:%s: %.1f%%", i, coco_labels_tab[(int32_t)nmsed_class_ptr[i]], nmsed_score_ptr[i] * 100);
        printf("\t(left: %4.0f   top: %4.0f   right: %4.0f   bottom: %4.0f)\n", nmsed_box_ptr[box_offset + 0],
               nmsed_box_ptr[box_offset + 1], nmsed_box_ptr[box_offset + 2], nmsed_box_ptr[box_offset + 3]);
        box_offset += 4;
    }
}

void YoloXOnnxTRTAPIExec(const string& model_path, const string& quant_param_path, const string& engine_path,
                         const string& image_path, const bool& with_qdq) {
    const string input_tensor_name{"images"}, output_num_detections{"num_detections"},
        output_detection_boxes{"detection_boxes"}, output_detection_scores{"detection_scores"},
        output_detection_classes{"detection_classes"};

    std::string input_name = input_tensor_name;
    std::vector<string> output_names = {output_num_detections, output_detection_boxes, output_detection_scores,
                                        output_detection_classes};

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
    if (with_qdq) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    } else {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
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
    ASSERT(num_input == 1);
    ASSERT(num_output == output_names.size());

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    ASSERT(inputDims.nbDims == 4);
    cout << "\nInput dimes: " << endl;
    for (auto i = 0; i < inputDims.nbDims; ++i) {
        cout << inputDims.d[i] << " ";
    }

    cout << "\nOutput dims: " << endl;
    for (int i = 0; i < num_output; ++i) {
        auto oup_dims = network->getOutput(i)->getDimensions();
        for (int j = 0; j < oup_dims.nbDims; ++j) {
            cout << oup_dims.d[j] << ",";
        }
        cout << endl;
    }

    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        return;
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    // Option operation, not necessary
    WriteBuffer2Disk(engine_path, plan->data(), plan->size());

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
    cout << "Input " << input_name << " index: " << input_idx << endl;
    std::unordered_map<std::string, int32_t> output_tensor_idx_map;
    for (const std::string& o_name : output_names) {
        auto output_idx = engine->getBindingIndex(o_name.c_str());
        cout << "Output " << o_name << " index: " << output_idx << endl;
        output_tensor_idx_map[o_name] = output_idx;
    }

    vector<float> image_mean = {0.f, 0.f, 0.f};
    vector<float> image_std = {1.f, 1.f, 1.f};
    bool convert_to_bgr = true;
    // auto cpu_fp32_image = LoadImageCPU(image_path, inputDims, true, image_mean, image_std, convert_to_bgr);
    std::vector<shared_ptr<float>> cpu_outputs;

    // allocate buffer
    std::vector<shared_ptr<float>> io_buffers_cpu;
    std::vector<void*> io_buffers_gpu(engine->getNbBindings());
    std::vector<void*> warmup_io_buffers_gpu(engine->getNbBindings());
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        int n_volume = volume(engine->getBindingDimensions(i));
        int n_bytes = GetBytes(engine->getBindingDimensions(i), engine->getBindingDataType(i));
        CHECK(cudaMalloc(&io_buffers_gpu.at(i), n_bytes));
        CHECK(cudaMalloc(&warmup_io_buffers_gpu.at(i), n_bytes));
        io_buffers_cpu.emplace_back(shared_ptr<float>(new float[n_volume], ArrayDeleter()));
        if (i == input_idx) {
            auto dims = Dims2Vec(engine->getBindingDimensions(i));
            LoadImageCPU(image_path, io_buffers_cpu.at(input_idx).get(), dims, 0, false, image_mean, image_std,
                         convert_to_bgr);
            CHECK(cudaMemcpy(io_buffers_gpu.at(input_idx), io_buffers_cpu.at(input_idx).get(), n_bytes,
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(warmup_io_buffers_gpu.at(input_idx), io_buffers_cpu.at(input_idx).get(), n_bytes,
                             cudaMemcpyHostToDevice));
        } else {
        }
    }
    cout << "User input date prepare done" << endl;

    auto context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }

    // Warmup
    for (auto i = 0; i < 50; ++i) {
        context->executeV2(warmup_io_buffers_gpu.data());
    }
    std::cout << "Warm up done" << endl;

    // decl event
    cudaEvent_t start, stop;
    float time;
    // create event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    auto status = context->executeV2(io_buffers_gpu.data());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "time: " << time / 1 << "ms" << std::endl;

    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    for (int i = 0; i < num_output; ++i) {
        int32_t tensor_idx = output_tensor_idx_map.at(output_names.at(i));
        int n_bytes = GetBytes(engine->getBindingDimensions(tensor_idx), engine->getBindingDataType(tensor_idx));
        CHECK(cudaMemcpy(io_buffers_cpu.at(tensor_idx).get(), io_buffers_gpu.at(tensor_idx), n_bytes,
                         cudaMemcpyDeviceToHost));
    }

    float* count_ptr = io_buffers_cpu.at(output_tensor_idx_map.at(output_num_detections)).get();
    float* nmsed_box_ptr = io_buffers_cpu.at(output_tensor_idx_map.at(output_detection_boxes)).get();
    float* nmsed_score_ptr = io_buffers_cpu.at(output_tensor_idx_map.at(output_detection_scores)).get();
    float* nmsed_class_ptr = io_buffers_cpu.at(output_tensor_idx_map.at(output_detection_classes)).get();
    PrintYoloXResult(count_ptr, nmsed_box_ptr, nmsed_score_ptr, nmsed_class_ptr);

    for (void* ptr : io_buffers_gpu) cudaFree(ptr);
    for (void* ptr : warmup_io_buffers_gpu) cudaFree(ptr);
    cout << "Yolox with IxRT API demo for enqueue done" << endl;
}

int main(int argc, char* argv[]) {
    CLI::App app{"App description"};

    std::string onnx_path = "data/yolox_m/yolox_m_with_decoder_nms.onnx";
    std::string quant_param_path = "";
    std::string image_path = "data/yolox_m/dog_640.jpg";
    std::string engine_path = "data/yolox_m/yolox_m_ixrt_0614.engine";
    std::string demo = "trt_exec";
    std::string mode = "without_qdq";
    std::string plugin_lib_path = "";
    app.add_option("--onnx", onnx_path, "YoloX onnx model path");
    app.add_option("-e, --engine", engine_path, "YoloX engine save path");
    app.add_option("-i, --input", image_path, "Test image");
    app.add_option("--demo", demo, "Choose which demo to run, available: old/trt_exec");
    app.add_option("-t, --type", mode, "The type of onnx model, available: without_qdq/with_qdq");
    app.add_option("-p, --plugin", plugin_lib_path, "Plugin lib path");
    CLI11_PARSE(app, argc, argv);

    if (not plugin_lib_path.empty()) {
        void* handle = dlopen(plugin_lib_path.c_str(), RTLD_LAZY);
        std::cout << "load plugin:" << plugin_lib_path << std::endl;
        if (handle == nullptr) {
            throw "Invaid plugin lib path " + plugin_lib_path;
        }
    } else {
        throw "Please pass plugin lib path.";
    }

    bool with_qdq = false;
    if (mode == "with_qdq") {
        with_qdq = true;
    } else if (mode == "without_qdq") {
        with_qdq = false;
    } else {
        throw "No such type called " + mode + ", available: (without_qdq/with_qdq)";
    }
    if (demo == "trt_exec") {
        YoloXOnnxTRTAPIExec(onnx_path, quant_param_path, engine_path, image_path, with_qdq);
    } else {
        throw "No such demo called " + demo + ", available: (old/trt_exec)";
    }

    return 0;
}
