#include <iostream>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"

using namespace nvinfer1;
using namespace nvinfer1::samples::common;

int main() {
    nvinfer1::samples::common::Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = UPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    auto cond = network->addInput("cond", DataType::kBOOL, Dims{0});
    auto x = network->addInput("x", DataType::kFLOAT, Dims{1, {9}});
    auto y = network->addInput("y", DataType::kFLOAT, Dims{1, {9}});

    auto condition_marker = network->addIfConditional();
    condition_marker->setName("condition_marker");
    condition_marker->setCondition(*cond);

    // Add input layers to demarcate entry into true/false branches.
    IIfConditionalInputLayer* condition_in_x = condition_marker->addInput(*x);
    auto in_x = condition_in_x->getOutput(0);
    IIfConditionalInputLayer* condition_in_y = condition_marker->addInput(*y);
    auto in_y = condition_in_y->getOutput(0);

    // true subgraph
    auto* true_subgraph = network->addElementWise(*in_x, *in_y, ElementWiseOperation::kSUM)->getOutput(0);

    // false subgraph
    auto* sub_out_tensor = network->addElementWise(*in_x, *in_y, ElementWiseOperation::kSUB)->getOutput(0);

    // nest false subgraph
    auto nest_cond = network->addInput("nest_cond", DataType::kBOOL, Dims{0});
    auto z = network->addInput("z", DataType::kFLOAT, Dims{1, {9}});

    auto nest_condition_marker = network->addIfConditional();
    nest_condition_marker->setName("nest_condition_marker");
    nest_condition_marker->setCondition(*nest_cond);

    IIfConditionalInputLayer* nest_condition_in_x = nest_condition_marker->addInput(*sub_out_tensor);
    auto nest_in_x = nest_condition_in_x->getOutput(0);
    IIfConditionalInputLayer* nest_condition_in_z = nest_condition_marker->addInput(*z);
    auto nest_in_z = nest_condition_in_z->getOutput(0);

    auto* true_nest_subgraph =
        network->addElementWise(*nest_in_x, *nest_in_z, ElementWiseOperation::kSUM)->getOutput(0);
    auto* false_nest_subgraph =
        network->addElementWise(*nest_in_x, *nest_in_z, ElementWiseOperation::kSUB)->getOutput(0);
    IIfConditionalOutputLayer* nest_condition_output =
        nest_condition_marker->addOutput(*true_nest_subgraph, *false_nest_subgraph);
    auto* false_subgraph = nest_condition_output->getOutput(0);

    IIfConditionalOutputLayer* condition_output = condition_marker->addOutput(*true_subgraph, *false_subgraph);
    auto* output = condition_output->getOutput(0);
    network->markOutput(*output);

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

    float cpu_x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float cpu_y[9] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    float cpu_z[9] = {3, 4, 5, 6, 7, 8, 9, 10, 11};

    void* gpu_x{nullptr};
    void* gpu_y{nullptr};
    void* gpu_z{nullptr};

    cudaMalloc(&gpu_x, sizeof(cpu_x));
    cudaMalloc(&gpu_y, sizeof(cpu_y));
    cudaMalloc(&gpu_z, sizeof(cpu_z));

    cudaMemcpy(gpu_x, cpu_x, sizeof(cpu_x), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, cpu_y, sizeof(cpu_y), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_z, cpu_z, sizeof(cpu_z), cudaMemcpyHostToDevice);

    bool cpu_cond1, cpu_cond2;
    void* gpu_cond1{nullptr};
    void* gpu_cond2{nullptr};
    cudaMalloc(&gpu_cond1, sizeof(cpu_cond1));
    cudaMalloc(&gpu_cond2, sizeof(cpu_cond2));

    std::array<float, 9> cpu_output;
    size_t cpu_output_size = cpu_output.size() * sizeof(float);

    void* output_gpu{nullptr};
    cudaMalloc(&output_gpu, cpu_output_size);

    // if_cond1 = true  if_cond2 = true
    cpu_cond1 = true;
    cpu_cond2 = true;
    cudaMemcpy(gpu_cond1, &cpu_cond1, sizeof(cpu_cond1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cond2, &cpu_cond2, sizeof(cpu_cond2), cudaMemcpyHostToDevice);

    void* binding_buffer3[] = {gpu_cond1, gpu_x, gpu_y, gpu_cond2, gpu_z, output_gpu};
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    context->executeV2(binding_buffer3);
    cudaMemcpy(cpu_output.data(), output_gpu, cpu_output_size, cudaMemcpyDeviceToHost);

    if (cpu_output != std::array<float, 9>{3, 5, 7, 9, 11, 13, 15, 17, 19}) {
        std::cout << "FAIL" << std::endl;
    } else {
        std::cout << "PASS" << std::endl;
    }

    cudaFree(gpu_cond1);
    cudaFree(gpu_cond2);
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);
    cudaFree(output_gpu);
    return 0;
}