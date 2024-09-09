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


#include <unistd.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "hide_data_transfer.h"
#include "image_io.h"
#include "logging.h"
#include "memory_utils.h"
#include "misc.h"
#include "pipeline.h"
#include "postprocess_utils.h"
#include "resnet18.h"

using std::cerr;
using std::cout;
using std::endl;

Resnet18* LoadModel(const std::string& model_path, std::string& quant_file, std::string& input_name,
                    std::string& output_name) {
    Resnet18* model = new Resnet18(model_path, quant_file, input_name, output_name);
    return model;
}

void IxRTAPIEnqueueHideDataTransferCPM() {
    std::string dir_path("data/resnet18/");
    std::string image_path(dir_path + "kitten_224.bmp");
    std::string model_path(dir_path + "resnet18.onnx");
    std::string quant_file(dir_path + "");
    std::string input_name("input");
    std::string output_name("output");

    auto load_method = std::bind(LoadModel, model_path, quant_file, input_name, output_name);
    Pipeline infer_pipeline;
    infer_pipeline.SetDeviceID(0);
    infer_pipeline.Start(load_method);

    std::vector<int32_t> input_dims{1, 3, 224, 224};
    int32_t input_n_volume = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int32_t>{});

    int32_t input_n_bytes = input_n_volume * sizeof(float);
    auto input_buffer = std::shared_ptr<float>(new float[input_n_volume], ArrayDeleter());

    int32_t online_infer_num = 4;
    for (auto i = 0; i < online_infer_num; ++i) {
        LoadImageCPU(image_path, input_buffer.get(), input_dims, 0);
        Tensor input(input_buffer.get(), input_n_bytes);
        cout << "\n"
             << i << "th"
             << " commit input" << endl;
        infer_pipeline.CommitInput(&input);
    }

    infer_pipeline.Stop();
}
