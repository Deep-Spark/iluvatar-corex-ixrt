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


#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "postprocess_utils.h"

using std::cerr;
using std::cout;
using std::endl;

std::string dir_path("data/resnet18/");
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

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weight_map;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weight_map[name] = wt;
    }

    return weight_map;
}

nvinfer1::IActivationLayer* basicBlock(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weight_map, nvinfer1::ITensor& input,
                                       int inch, int outch, int stride, std::string lname) {
    // nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(
        input, outch, nvinfer1::Dims{2, {3, 3}}, weight_map[lname + "conv1.weight"], weight_map[lname + "conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
    conv1->setPaddingNd(nvinfer1::Dims{2, {1, 1}});

    nvinfer1::IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IConvolutionLayer* conv2 =
        network->addConvolutionNd(*relu1->getOutput(0), outch, nvinfer1::Dims{2, {3, 3}},
                                  weight_map[lname + "conv2.weight"], weight_map[lname + "conv2.bias"]);
    assert(conv2);
    conv2->setPaddingNd(nvinfer1::Dims{2, {1, 1}});

    nvinfer1::IElementWiseLayer* ew1;
    if (inch != outch) {
        nvinfer1::IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, nvinfer1::Dims{2, {1, 1}},
                                                                       weight_map[lname + "downsample.0.weight"],
                                                                       weight_map[lname + "downsample.0.bias"]);
        assert(conv3);
        conv3->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
        ew1 = network->addElementWise(*conv3->getOutput(0), *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    }
    nvinfer1::IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

void ConstructNetwork(nvinfer1::INetworkDefinition* network, const std::string& input_name,
                      const std::string& output_name, const std::string& weight_path) {
    std::map<std::string, nvinfer1::Weights> weight_map = loadWeights(weight_path);
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ITensor* data =
        network->addInput(input_name.c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::Dims{4, {1, 3, 224, 224}});
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(
        *data, 64, nvinfer1::Dims{2, {7, 7}}, weight_map["conv1.weight"], weight_map["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::Dims{2, {2, 2}});
    conv1->setPaddingNd(nvinfer1::Dims{2, {3, 3}});

    nvinfer1::IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);

    nvinfer1::IPoolingLayer* pool1 =
        network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::Dims{2, {3, 3}});
    assert(pool1);
    pool1->setStrideNd(nvinfer1::Dims{2, {2, 2}});
    pool1->setPaddingNd(nvinfer1::Dims{2, {1, 1}});

    nvinfer1::IActivationLayer* relu2 = basicBlock(network, weight_map, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    nvinfer1::IActivationLayer* relu3 = basicBlock(network, weight_map, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");
    nvinfer1::IActivationLayer* relu4 = basicBlock(network, weight_map, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    nvinfer1::IActivationLayer* relu5 = basicBlock(network, weight_map, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");
    nvinfer1::IActivationLayer* relu6 = basicBlock(network, weight_map, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    nvinfer1::IActivationLayer* relu7 = basicBlock(network, weight_map, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");
    nvinfer1::IActivationLayer* relu8 = basicBlock(network, weight_map, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    nvinfer1::IActivationLayer* relu9 = basicBlock(network, weight_map, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    nvinfer1::IPoolingLayer* pool2 =
        network->addPoolingNd(*relu9->getOutput(0), nvinfer1::PoolingType::kAVERAGE, nvinfer1::Dims{2, {7, 7}});
    assert(pool2);
    pool2->setStrideNd(nvinfer1::Dims{2, {1, 1}});

    nvinfer1::IShuffleLayer* flatten = network->addShuffle(*pool2->getOutput(0));
    assert(flatten);
    flatten->setReshapeDimensions(nvinfer1::Dims{2, {1, 512}});

    nvinfer1::IConstantLayer* gemm_weight =
        network->addConstant(nvinfer1::Dims{2, {1000, 512}}, weight_map["fc.weight"]);
    nvinfer1::IMatrixMultiplyLayer* matmul =
        network->addMatrixMultiply(*flatten->getOutput(0), nvinfer1::MatrixOperation::kNONE, *gemm_weight->getOutput(0),
                                   nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(matmul);

    nvinfer1::IConstantLayer* gemm_bias = network->addConstant(nvinfer1::Dims{2, {1, 1000}}, weight_map["fc.bias"]);
    nvinfer1::IElementWiseLayer* gemm_bias_add =
        network->addElementWise(*matmul->getOutput(0), *gemm_bias->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(gemm_bias_add);
    network->markOutput(*gemm_bias_add->getOutput(0));
    PrintDims(gemm_bias_add->getOutput(0)->getDimensions(), "gemm output dim: ");
}

void IxRTAPIConstructModel() {
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string weight_path(dir_path + "resnet18_fusebn.wts");
    std::string input_name("input");
    std::string output_name("ElementWise_51_out");
    std::string engine_save_path(dir_path + "api_construct_resnet18.engine");
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

    ConstructNetwork(network.get(), input_name, output_name, weight_path);
    cout << "layer number: " << network->getNbLayers() << endl;
    cout << "first layer: " << endl;
    nvinfer1::ILayer* lyr = network->getLayer(0);
    cout << "layer name: " << lyr->getName() << endl;
    for (int j = 0; j < lyr->getNbInputs(); j++) {
        std::string in_name = lyr->getInput(j)->getName();
        cout << "in tensor name: " << in_name << endl;
    }

    for (int j = 0; j < lyr->getNbOutputs(); j++) {
        std::string out_name = lyr->getOutput(j)->getName();
        cout << "out tensor name: " << out_name << endl;
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
    for (auto i = 0; i < outputDims.nbDims; ++i) {
        cout << outputDims.d[i] << " ";
    }
    cout << endl;

    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (not config) {
        std::cout << "Create config failed" << std::endl;
        return;
    } else {
        cout << "Create config success" << endl;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (not plan) {
        std::cout << "Create serialized engine plan failed" << std::endl;
        return;
    } else {
        cout << "Create serialized engine plan done" << endl;
    }

    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (not runtime) {
        std::cout << "Create runtime failed" << std::endl;
        return;
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
    inputDims = engine->getBindingDimensions(input_idx);
    outputDims = engine->getBindingDimensions(output_idx);

    auto cpu_fp32_image = LoadImageCPU(image_path, inputDims);
    std::vector<void*> binding_buffer(engine->getNbBindings());
    auto input_size = GetBytes(inputDims, engine->getBindingDataType(input_idx));
    auto output_size = GetBytes(outputDims, engine->getBindingDataType(output_idx));
    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    void* input_gpu{nullptr};
    CHECK(cudaMalloc(&input_gpu, input_size));
    void* output_gpu{nullptr};
    CHECK(cudaMalloc(&output_gpu, output_size));
    CHECK(cudaMemcpy(input_gpu, cpu_fp32_image.get(), input_size, cudaMemcpyHostToDevice));
    cout << "User input date prepare done" << endl;
    binding_buffer.at(input_idx) = input_gpu;
    binding_buffer.at(output_idx) = output_gpu;
    auto context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context) {
        cout << "Create execution context done" << endl;
    } else {
        cout << "Create execution context failed" << endl;
    }
    // Warmup
    for (auto i = 0; i < 20; ++i) {
        context->executeV2(binding_buffer.data());
    }
    auto start = NowUs();
    auto status = context->executeV2(binding_buffer.data());
    uint64_t time = NowUs() - start;
    if (not status) {
        cerr << "Execute ixrt failed" << endl;
    } else {
        cout << "Execute ixrt success" << endl;
    }

    CHECK(cudaMemcpy(cpu_output.get(), output_gpu, output_size, cudaMemcpyDeviceToHost));
    std::cout << "out size: " << output_size / sizeof(float) << std::endl;
    GetClassificationResult(cpu_output.get(), 1000, 5, 0);

    CHECK(cudaFree(input_gpu));
    CHECK(cudaFree(output_gpu));
    cout << "Construct resnet18 with IxRT API demo done" << endl;

    float fps = 1 / ((float)time / 1000000);
    std::cout << "BatchSize: " << 1 << " FPS: " << fps << std::endl;
}

int main(int argc, char* argv[]) {
    IxRTAPIConstructModel();

    return 0;
}
