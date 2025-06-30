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

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <cstdint>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "nmsKernel.h"

namespace nvinfer1::plugin {

//!
//! \brief The NMSParameters are used by the NMSPlugin for performing
//! the non_max_suppression operation over boxes for object detection networks.
//! \param share_location If set to true, the boxes inputs are shared across all
//!        classes. If set to false, the boxes input should account for per class box data.
//! \param background_class Label ID for the background class. If there is no background class, set it as -1
//! \param max_output_boxes Number of total bounding boxes to be kept per image after NMS step.
//! \param score_threshold Scalar threshold for score (low scoring boxes are removed).
//! \param iou_threshold scalar threshold for IOU (new boxes that have high IOU overlap
//!
struct NMSParameters {
    bool share_location;
    int32_t max_output_boxes, background_class;
    float score_threshold, iou_threshold;
};

class NMS : public IPluginV2DynamicExt {
   public:
    NMS(NMSParameters params);
    NMS(void const* data, size_t length);
    NMS() noexcept = delete;
    ~NMS() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

   private:
    void checkValidInputs(DynamicPluginTensorDesc const* inputs, int32_t nbInputDims);
    void validateAttributes(NMSParameters params);

    NMSParameters mParams{};

    int32_t mMaxThreadsPerBlock{};

    std::string mNameSpace{};
};

class NMSPluginCreator : public IPluginCreator {
   public:
    NMSPluginCreator();

    ~NMSPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(char const* name, void const* serialData,
                                           size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

   private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

}  // namespace nvinfer1::plugin
