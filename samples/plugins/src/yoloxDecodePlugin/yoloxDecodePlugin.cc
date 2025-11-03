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

#include "yoloxDecodePlugin.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "PluginCheckMacros.h"
#include "common_def.cuh"
#include "plugin.h"
#include "yoloxDecodeKernel.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::YoloxDecoder;
using nvinfer1::plugin::YoloxDecoderPluginCreator;

namespace {
char const* kYOLOXDECODER_PLUGIN_VERSION{"1"};
char const* kYOLOXDECODER_PLUGIN_NAME{"YoloxDecoder_IXRT"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(int32_t) * 1 + sizeof(int32_t) * 1 + sizeof(int32_t) * 2};
}  // namespace

PluginFieldCollection YoloxDecoderPluginCreator::mFC{};
std::vector<PluginField> YoloxDecoderPluginCreator::mPluginAttributes;

YoloxDecoderPluginCreator::YoloxDecoderPluginCreator() {
    printf("Call yolox plugin constructor here\n");
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* YoloxDecoderPluginCreator::getPluginName() const noexcept { return kYOLOXDECODER_PLUGIN_NAME; }

char const* YoloxDecoderPluginCreator::getPluginVersion() const noexcept { return kYOLOXDECODER_PLUGIN_VERSION; }

PluginFieldCollection const* YoloxDecoderPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* YoloxDecoderPluginCreator::createPlugin(char const* name,
                                                             PluginFieldCollection const* fc) noexcept {
    try {
        PLUGIN_ASSERT(fc != nullptr);
        PluginField const* fields = fc->fields;

        // default values
        int32_t num_class = 80;
        int32_t stride = 32;

        for (int32_t i = 0; i < fc->nbFields; ++i) {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "num_class")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                num_class = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));

            } else if (!strcmp(attrName, "stride")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                stride = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
        }
        IPluginV2DynamicExt* plugin = new YoloxDecoder(num_class, stride);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* YoloxDecoderPluginCreator::deserializePlugin(char const* name, void const* data,
                                                                  size_t length) noexcept {
    try {
        PLUGIN_ASSERT(data != nullptr);
        return new YoloxDecoder(data, length);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void YoloxDecoderPluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* YoloxDecoderPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

int32_t YoloxDecoder::getNbOutputs() const noexcept { return 2; }

int32_t YoloxDecoder::initialize() noexcept {
    int32_t device;
    PLUGIN_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    PLUGIN_CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    // mMaxThreadsPerBlock = props.maxThreadsPerBlock;
    mMaxThreadsPerBlock = kNbThreadsPerBlockGainBestPerformance;

    return 0;
}

void YoloxDecoder::terminate() noexcept {}

void YoloxDecoder::destroy() noexcept { delete this; }

size_t YoloxDecoder::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
                                      int32_t nbOutputs) const noexcept {
    return 0;
}

bool YoloxDecoder::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                             int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(pos >= 0 && pos <= 4);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR) {
        return false;
    }

    // first input should be float16 or float32
    if (pos == 0) {
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
    }

    // the output should have the same type as the first input
    return (inOut[pos].type == inOut[0].type);
}

char const* YoloxDecoder::getPluginType() const noexcept { return kYOLOXDECODER_PLUGIN_NAME; }

char const* YoloxDecoder::getPluginVersion() const noexcept { return kYOLOXDECODER_PLUGIN_VERSION; }

IPluginV2DynamicExt* YoloxDecoder::clone() const noexcept {
    try {
        auto plugin = new YoloxDecoder(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void YoloxDecoder::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* YoloxDecoder::getPluginNamespace() const noexcept { return mNameSpace.c_str(); }

void YoloxDecoder::checkValidInputs(DynamicPluginTensorDesc const* inputs, int32_t nbInputDims) {
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputDims == 3);

    Dims input0_dims = inputs[0].desc.dims;
    Dims input1_dims = inputs[1].desc.dims;
    Dims input2_dims = inputs[2].desc.dims;
    PLUGIN_ASSERT(input0_dims.nbDims == 4);
    PLUGIN_ASSERT(input1_dims.nbDims == 4);
    PLUGIN_ASSERT(input2_dims.nbDims == 4);
}

void YoloxDecoder::validateAttributes(int32_t num_class, int32_t stride) {
    PLUGIN_ASSERT(num_class > 0);
    PLUGIN_ASSERT(stride % 2 == 0);
}

DimsExprs YoloxDecoder::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                            IExprBuilder& exprBuilder) noexcept {
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);

    DimsExprs result;

    if (outputIndex == 0) {
        result.nbDims = 4;
        // batch
        result.d[0] = inputs[1].d[0];
        // number_box_parameters
        result.d[1] = inputs[1].d[1];
        // candidate classes
        result.d[2] = exprBuilder.constant(1);
        // num_boxes
        result.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    } else {
        result.nbDims = 3;
        // batch
        result.d[0] = inputs[0].d[0];
        // number_classes
        result.d[1] = inputs[0].d[1];
        // num_boxes
        result.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    }

    return result;
}

int32_t YoloxDecoder::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                              void const* const* inputs, void* const* outputs, void* workspace,
                              cudaStream_t stream) noexcept {
    PLUGIN_ASSERT(inputDesc != nullptr);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(outputDesc != nullptr);

    int32_t batch = inputDesc[0].dims.d[0];

    auto type = inputDesc[0].type;

    PLUGIN_ASSERT(type == DataType::kHALF || type == DataType::kFLOAT);

    switch (type) {
        case DataType::kFLOAT: {
            return YoloxDecoderImpl<float>(stream, mMaxThreadsPerBlock, inputs[0], inputs[1], inputs[2], outputs[0],
                                           outputs[1], mNumClass, mStride, batch, mHeight, mWidth);
        } break;
        case DataType::kHALF: {
            return YoloxDecoderImpl<__half>(stream, mMaxThreadsPerBlock, inputs[0], inputs[1], inputs[2], outputs[0],
                                            outputs[1], mNumClass, mStride, batch, mHeight, mWidth);
        } break;
        default:
            return -1;
    }

    return 0;
}

size_t YoloxDecoder::getSerializationSize() const noexcept { return kSERIALIZATION_SIZE; }

void YoloxDecoder::serialize(void* buffer) const noexcept {
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write(d, mNumClass);  // int32_t
    write(d, mStride);    // int32_t
    write(d, mHeight);    // int32_t
    write(d, mWidth);     // int32_t
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

YoloxDecoder::YoloxDecoder(int32_t numClass, int32_t stride) : mNumClass(numClass), mStride(stride) {
    initialize();
    validateAttributes(numClass, stride);
}

YoloxDecoder::YoloxDecoder(void const* data, size_t length) {
    PLUGIN_ASSERT(data != nullptr);
    PLUGIN_ASSERT(length == kSERIALIZATION_SIZE);
    initialize();

    char const* d = static_cast<char const*>(data);
    char const* a = d;
    mNumClass = read<int32_t>(d);
    mStride = read<int32_t>(d);
    mHeight = read<int32_t>(d);
    mWidth = read<int32_t>(d);

    PLUGIN_ASSERT(d == a + length);

    validateAttributes(mNumClass, mStride);
}

DataType YoloxDecoder::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept {
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(index < 2);
    return inputTypes[0];
}

void YoloxDecoder::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                   DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbOutputs == 2);
    PLUGIN_ASSERT(nbInputs == 3);

    checkValidInputs(in, nbInputs);

    mHeight = in[0].desc.dims.d[2];
    mWidth = in[0].desc.dims.d[3];
}

REGISTER_TENSORRT_PLUGIN(YoloxDecoderPluginCreator);
