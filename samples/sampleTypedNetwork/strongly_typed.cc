#include <cstddef>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "typed_network.h"
namespace nvinfer1::samples {
using namespace common;
void StrongTypedNetworkSample() {
    Logger logger(nvinfer1::ILogger::Severity::kVERBOSE);
    auto builder = UPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    // 创建网络定义，启用强类型网络标志
    auto network = UPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    nvinfer1::Dims input_dims{4, {4, 1, 5, 5}};
    nvinfer1::Dims out_dims{4, {4, 1, 2, 2}};
    auto input_data = network->addInput("input_data", nvinfer1::DataType::kFLOAT, input_dims);
    std::array<float, 100> input_host_data;
    for (int32_t i = 0; i < 100; ++i) {
        input_host_data[i] = 1;
    }
    std::array<float, 9> kernel_3x3_data{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    std::array<float, 1> bias_1{0.2};
    std::array<float, 4> kernel_2x2_data{0.3, 0.3, 0.3, 0.3};
    std::array<float, 1> bias_2{0.2};
    std::array<float, 1> kernel_1x1_data{0.4};
    std::array<float, 1> bias_3{0.2};
    nvinfer1::Weights weights_1{nvinfer1::DataType::kFLOAT, kernel_3x3_data.data(), kernel_3x3_data.size()};
    nvinfer1::Weights weights_2{nvinfer1::DataType::kFLOAT, kernel_2x2_data.data(), kernel_2x2_data.size()};
    nvinfer1::Weights weights_3{nvinfer1::DataType::kFLOAT, kernel_1x1_data.data(), kernel_1x1_data.size()};
    nvinfer1::Weights bw_1{nvinfer1::DataType::kFLOAT, bias_1.data(), bias_1.size()};
    nvinfer1::Weights bw_2{nvinfer1::DataType::kFLOAT, bias_2.data(), bias_2.size()};
    nvinfer1::Weights bw_3{nvinfer1::DataType::kFLOAT, bias_3.data(), bias_3.size()};
    nvinfer1::Dims kernel_size_3x3{2, {3, 3}};
    nvinfer1::Dims kernel_size_2x2{2, {2, 2}};
    nvinfer1::Dims kernel_size_1x1{2, {1, 1}};
    auto conv_layer_1 = network->addConvolutionNd(*input_data, 1, kernel_size_3x3, weights_1, bw_1);
    conv_layer_1->setName("conv1");
    auto conv_1_output = conv_layer_1->getOutput(0);
    auto conv_layer_2 = network->addConvolutionNd(*conv_1_output, 1, kernel_size_2x2, weights_2, bw_2);
    conv_layer_2->setName("conv2");
    auto conv_2_output = conv_layer_2->getOutput(0);
    auto conv_layer_3 = network->addConvolutionNd(*conv_2_output, 1, kernel_size_1x1, weights_3, bw_3);
    auto conv_output_3 = conv_layer_3->getOutput(0);
    network->markOutput(*conv_output_3);

    // 创建构建器配置
    auto config = UPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // 构建序列化网络
    auto plan = UPtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    UPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    ASSERT(runtime);

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                         ObjectDeleter());
    ASSERT(engine);
    auto context = UPtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    ASSERT(context);
    size_t input_size = GetBytes(input_dims, nvinfer1::DataType::kFLOAT);
    size_t output_size = GetBytes(out_dims, nvinfer1::DataType::kFLOAT);
    std::vector<void*> binding_buffer(2, nullptr);
    CHECK(cudaMalloc(&binding_buffer[0], input_size));
    CHECK(cudaMalloc(&binding_buffer[1], output_size));
    cudaMemcpy(binding_buffer[0], input_host_data.data(), input_size, cudaMemcpyHostToDevice);

    std::shared_ptr<float> cpu_output(new float[output_size / sizeof(float)], ArrayDeleter());
    context->executeV2(binding_buffer.data());
    cudaMemcpy(cpu_output.get(), binding_buffer[1], output_size, cudaMemcpyDeviceToHost);
    bool result = true;
    for (int32_t i = 0; i < 16; ++i) {
        if (std::abs(cpu_output.get()[i] - 1.24) > 0.00001) {
            std::cout << cpu_output.get()[i] << std::endl;
            result = false;
            break;
        }
    }
    if (result) {
        std::cout << "Infer strongly type finish, pass" << std::endl;
    } else {
        std::cout << "Infer strongly type finish, fail" << std::endl;
    }
}
}  // namespace nvinfer1::samples
