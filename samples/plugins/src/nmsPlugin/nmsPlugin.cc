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

#include "nmsPlugin.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "PluginCheckMacros.h"
#include "common_def.cuh"
#include "nmsImpl.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::NMS;
using nvinfer1::plugin::NMSPluginCreator;

namespace {
char const* kNMS_PLUGIN_VERSION{"1"};
char const* kNMS_PLUGIN_NAME{"NMS_IXRT"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(NMSParameters)};
}  // namespace

PluginFieldCollection NMSPluginCreator::mFC{};
std::vector<PluginField> NMSPluginCreator::mPluginAttributes;

NMSPluginCreator::NMSPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("share_location", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("background_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* NMSPluginCreator::getPluginName() const noexcept { return kNMS_PLUGIN_NAME; }

char const* NMSPluginCreator::getPluginVersion() const noexcept { return kNMS_PLUGIN_VERSION; }

PluginFieldCollection const* NMSPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* NMSPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        PLUGIN_ASSERT(fc != nullptr);
        PluginField const* fields = fc->fields;

        // default values
        NMSParameters params;

        for (int32_t i = 0; i < fc->nbFields; ++i) {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "share_location")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.share_location = *(static_cast<const bool*>(fields[i].data));
            } else if (!strcmp(attrName, "iou_threshold")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.iou_threshold = *(static_cast<const float*>(fields[i].data));
            } else if (!strcmp(attrName, "score_threshold")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.score_threshold = *(static_cast<const float*>(fields[i].data));
            } else if (!strcmp(attrName, "max_output_boxes")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.max_output_boxes = *(static_cast<const int32_t*>(fields[i].data));
            } else if (!strcmp(attrName, "background_class")) {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.background_class = *(static_cast<const int32_t*>(fields[i].data));
            }
        }
        IPluginV2DynamicExt* plugin = new NMS(params);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* NMSPluginCreator::deserializePlugin(char const* name, void const* data, size_t length) noexcept {
    try {
        PLUGIN_ASSERT(data != nullptr);
        return new NMS(data, length);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void NMSPluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* NMSPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);

int32_t NMS::getNbOutputs() const noexcept { return 4; }

int32_t NMS::initialize() noexcept {
    int32_t device;
    PLUGIN_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    PLUGIN_CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    // mMaxThreadsPerBlock = props.maxThreadsPerBlock;
    mMaxThreadsPerBlock = kNbThreadsPerBlockGainBestPerformance;

    return 0;
}

void NMS::terminate() noexcept {}

void NMS::destroy() noexcept { delete this; }

size_t NMS::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
                             int32_t nbOutputs) const noexcept {
    int32_t batch_size = inputs[0].dims.d[0];

    int32_t boxes_size = inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
    int32_t scores_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];

    int32_t num_classes = inputs[1].dims.d[1];
    int32_t num_priors = inputs[1].dims.d[2];

    auto type = inputs[0].type;
    PLUGIN_ASSERT(type == DataType::kHALF || type == DataType::kFLOAT);

    int top_k = 2048;
    switch (type) {
        case DataType::kFLOAT: {
            return detectionInferenceWorkspaceSize<float>(mParams.share_location, batch_size, boxes_size, scores_size,
                                                          num_classes, num_priors, top_k);
        } break;
        case DataType::kHALF: {
            return detectionInferenceWorkspaceSize<__half>(mParams.share_location, batch_size, boxes_size, scores_size,
                                                           num_classes, num_priors, top_k);
        } break;
        default:
            return -1;
    }
}

bool NMS::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                    int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(pos >= 0 && pos <= 5);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 4);

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

char const* NMS::getPluginType() const noexcept { return kNMS_PLUGIN_NAME; }

char const* NMS::getPluginVersion() const noexcept { return kNMS_PLUGIN_VERSION; }

IPluginV2DynamicExt* NMS::clone() const noexcept {
    try {
        auto plugin = new NMS(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

void NMS::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* NMS::getPluginNamespace() const noexcept { return mNameSpace.c_str(); }

void NMS::checkValidInputs(DynamicPluginTensorDesc const* inputs, int32_t nbInputDims) {
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputDims == 2);

    Dims input0_dims = inputs[0].desc.dims;
    Dims input1_dims = inputs[1].desc.dims;
    PLUGIN_ASSERT(input0_dims.nbDims == 4);
    PLUGIN_ASSERT(input1_dims.nbDims == 3);
}

void NMS::validateAttributes(NMSParameters params) { PLUGIN_ASSERT(params.share_location == true); }

DimsExprs NMS::getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                   IExprBuilder& exprBuilder) noexcept {
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(outputIndex < 4);  // there are four outputs.

    DimsExprs result;

    if (outputIndex == 0) {
        result.nbDims = 2;
        result.d[0] = inputs[0].d[0];
        result.d[1] = exprBuilder.constant(1);
    } else if (outputIndex == 1) {
        result.nbDims = 3;
        result.d[0] = inputs[0].d[0];
        result.d[1] = exprBuilder.constant(mParams.max_output_boxes);
        result.d[2] = exprBuilder.constant(4);
    } else {
        result.nbDims = 2;
        result.d[0] = inputs[0].d[0];
        result.d[1] = exprBuilder.constant(mParams.max_output_boxes);
    }

    return result;
}

int32_t NMS::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
                     void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    PLUGIN_ASSERT(inputDesc != nullptr);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(outputDesc != nullptr);

    int32_t batch_size = inputDesc[0].dims.d[0];

    int32_t boxes_size = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    int32_t scores_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];

    int32_t num_classes = inputDesc[1].dims.d[1];
    int32_t num_priors = inputDesc[1].dims.d[2];

    auto type = inputDesc[0].type;
    PLUGIN_ASSERT(type == DataType::kHALF || type == DataType::kFLOAT);

    switch (type) {
        case DataType::kFLOAT: {
            return NMSImpl<float>(stream, mMaxThreadsPerBlock, batch_size, boxes_size, scores_size,
                                  mParams.share_location, mParams.background_class, num_priors, num_classes,
                                  mParams.max_output_boxes, mParams.score_threshold, mParams.iou_threshold, inputs[0],
                                  inputs[1], outputs[0], outputs[1], outputs[2], outputs[3], workspace);
        } break;
        case DataType::kHALF: {
            return NMSImpl<__half>(stream, mMaxThreadsPerBlock, batch_size, boxes_size, scores_size,
                                   mParams.share_location, mParams.background_class, num_priors, num_classes,
                                   mParams.max_output_boxes, mParams.score_threshold, mParams.iou_threshold, inputs[0],
                                   inputs[1], outputs[0], outputs[1], outputs[2], outputs[3], workspace);
        } break;
        default:
            return -1;
    }

    return 0;
}

size_t NMS::getSerializationSize() const noexcept { return kSERIALIZATION_SIZE; }

void NMS::serialize(void* buffer) const noexcept {
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write(d, mParams);  // NMSParameters

    PLUGIN_ASSERT(d == a + getSerializationSize());
}

NMS::NMS(NMSParameters params) : mParams(params) { validateAttributes(params); }

NMS::NMS(void const* data, size_t length) {
    PLUGIN_ASSERT(data != nullptr);
    PLUGIN_ASSERT(length == kSERIALIZATION_SIZE);

    char const* d = static_cast<char const*>(data);
    char const* a = d;
    mParams = read<NMSParameters>(d);

    PLUGIN_ASSERT(d == a + length);

    validateAttributes(mParams);
}

DataType NMS::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept {
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == 2);
    return inputTypes[0];
}

void NMS::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbOutputs == 4);
    PLUGIN_ASSERT(nbInputs == 2);

    checkValidInputs(in, nbInputs);
}
