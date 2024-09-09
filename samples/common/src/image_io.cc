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


#include "image_io.h"

#include <iostream>
#include <ostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <cstring>
#include <numeric>

#include "stb_image_resize.h"
using namespace nvinfer1;
// image channel first CHW
void ImageRGB2BGR(int w, int h, uint8_t* data) {
    int i;
    for (i = 0; i < w * h; ++i) {
        uint8_t swap = data[3 * i];
        data[3 * i] = data[3 * i + 2];
        data[3 * i + 2] = swap;
    }
}

std::shared_ptr<uint8_t> GetImagePtr(const std::string& file_name, int32_t* w, int32_t* h, int32_t* c, int32_t mode) {
    uint8_t* raw_ptr = stbi_load(file_name.c_str(), w, h, c, mode);
    std::shared_ptr<uint8_t> raw_data(raw_ptr, [](uint8_t* p) { stbi_image_free(p); });
    return raw_data;
}

#ifdef USE_OPENCV_API
#include <opencv2/opencv.hpp>
int32_t ImagePreprocess(const cv::Mat& src_img, float* in_data) {
    const int32_t height = src_img.rows;
    const int32_t width = src_img.cols;
    const int32_t channels = src_img.channels();

    // convert data from bgr/gray to rgb
    cv::Mat rgb_img;
    if (channels == 3) {
        cvtColor(src_img, rgb_img, cv::COLOR_BGR2RGB);
    } else if (channels == 1) {
        cvtColor(src_img, rgb_img, cv::COLOR_GRAY2RGB);
    } else {
        fprintf(stderr, "unsupported channel num: %d\n", channels);
        return -1;
    }

    // split 3 channel to change HWC to CHW
    std::vector<cv::Mat> rgb_channels(3);
    split(rgb_img, rgb_channels);

    // by this constructor, when cv::Mat r_channel_fp32 changed, in_data will
    // also change
    cv::Mat r_channel_fp32(height, width, CV_32FC1, in_data + 0 * height * width);
    cv::Mat g_channel_fp32(height, width, CV_32FC1, in_data + 1 * height * width);
    cv::Mat b_channel_fp32(height, width, CV_32FC1, in_data + 2 * height * width);
    std::vector<cv::Mat> rgb_channels_fp32{r_channel_fp32, g_channel_fp32, b_channel_fp32};

    // convert uint8 to fp32, y = (x - mean) / std
    const float mean[3] = {0, 0, 0};  // change mean & std according to your dataset & training param
    const float std[3] = {255.0f, 255.0f, 255.0f};
    for (uint32_t i = 0; i < rgb_channels.size(); ++i) {
        rgb_channels[i].convertTo(rgb_channels_fp32[i], CV_32FC1, 1.0f / std[i], -mean[i] / std[i]);
    }

    return 0;
}
#endif

template <typename BufferType>
void LoadImageCPU(const std::string& file_name, BufferType* image, const std::vector<int32_t>& tensor_shape,
                  int32_t batch_idx, bool use_hwc, const std::vector<float>& mean, const std::vector<float>& std,
                  bool convert_to_bgr) {
    float* cpu_data_ptr = reinterpret_cast<float*>(image->GetDataPtr());
    LoadImageCPU(file_name, cpu_data_ptr, tensor_shape, batch_idx, use_hwc, mean, std, convert_to_bgr);
}

template <>
void LoadImageCPU(const std::string& file_name, float* image, const std::vector<int32_t>& tensor_shape,
                  int32_t batch_idx, bool use_hwc, const std::vector<float>& mean, const std::vector<float>& std,
                  bool convert_to_bgr) {
    if (batch_idx >= tensor_shape.at(0)) {
        std::cerr << "Set image data for batch idx: " << batch_idx << ", out of boundary: " << tensor_shape.at(0)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::shared_ptr<uint8_t> image_ptr{nullptr};
    int32_t w, h, c;
    image_ptr = GetImagePtr(file_name, &w, &h, &c, 0);
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (tensor_shape.at(2) != h or tensor_shape.at(3) != w) {
        std::cerr << "Image preprocess not include resize, image raw size must "
                     "be same with input setting "
                  << ", image h: " << h << ", w: " << w << " memory h: " << tensor_shape.at(2)
                  << ", w: " << tensor_shape.at(3) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (convert_to_bgr and c != 3) {
        std::cerr << "The channel of image: " << c << ", can not convert to BGR" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    uint64_t num_element = std::accumulate(tensor_shape.begin() + 1, tensor_shape.end(), 1, std::multiplies<int32_t>());

    float* cpu_data_ptr = image + batch_idx * num_element;
    uint8_t* raw_image_ptr = image_ptr.get();
    std::memset(cpu_data_ptr, 0.f, sizeof(float) * num_element);
    // Only set first image content
    uint64_t hw = w * h;
    for (int hw_id = 0; hw_id < hw; hw_id++) {
        for (int c_id = 0; c_id < c; c_id++) {
            int dst_c_id = convert_to_bgr ? 2 - c_id : c_id;
            int dst_id = use_hwc ? hw_id * c + dst_c_id : dst_c_id * hw + hw_id;
            cpu_data_ptr[dst_id] = (static_cast<float>(*raw_image_ptr) - mean[c_id]) / std[c_id];
            raw_image_ptr++;
        }
    }
}

std::shared_ptr<float> LoadImageCPU(const std::string& file_name, const nvinfer1::Dims& dims, bool chw) {
    std::shared_ptr<uint8_t> image_ptr{nullptr};
    int32_t w, h, c;
    image_ptr = GetImagePtr(file_name, &w, &h, &c, 0);
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    //    ImageRGB2BGR(w, h, image_ptr.get());

    if (dims.d[2] != h or dims.d[3] != w) {
        std::cerr << "Image preprocess not include resize, image raw size must "
                     "be same with input setting "
                  << ", image h: " << h << ", w: " << w << " memory h: " << dims.d[2] << ", w: " << dims.d[3]
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    uint64_t num_element = w * h * c;

    std::shared_ptr<float> fp32_data(new float[num_element], [](float* p) { delete[] p; });
    auto cpu_data_ptr = fp32_data.get();
    uint8_t* raw_image_ptr = image_ptr.get();
    // Raw to float for IxRT API
    // Only set first image content
    if (chw) {
        uint64_t raw_data_area = w * h;
        // From HWC to CHW for real IxRT API
        for (auto i = 0; i < raw_data_area; ++i) {
            for (auto k = 0; k < c; ++k) {
                cpu_data_ptr[i + k * raw_data_area] = static_cast<float>(raw_image_ptr[i * 3 + k]) / 255.f;
            }
        }
    } else {
        for (auto i = 0; i < num_element; ++i) {
            *cpu_data_ptr = static_cast<float>(*raw_image_ptr) / 255.f;
            cpu_data_ptr++;
            raw_image_ptr++;
        }
    }

    return fp32_data;
}

std::shared_ptr<float> LoadNormalizeImageCPU(const std::string& file_name, const nvinfer1::Dims& dims,
                                             const std::vector<float>& mean, const std::vector<float>& std) {
    std::shared_ptr<uint8_t> image_ptr{nullptr};
    int32_t w, h, c;
    image_ptr = GetImagePtr(file_name, &w, &h, &c, 0);
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    //    ImageRGB2BGR(w, h, image_ptr.get());

    if (dims.d[2] != h or dims.d[3] != w) {
        std::cerr << "Image preprocess not include resize, image raw size must "
                     "be same with input setting "
                  << ", image h: " << h << ", w: " << w << " memory h: " << dims.d[2] << ", w: " << dims.d[3]
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    uint64_t num_element = w * h * c;

    std::shared_ptr<float> fp32_data(new float[num_element], [](float* p) { delete[] p; });
    auto cpu_data_ptr = fp32_data.get();
    uint8_t* raw_image_ptr = image_ptr.get();
    // Raw to float for IxRT API
    // Only set first image content
    uint64_t raw_data_area = w * h;
    // From HWC to CHW for real IxRT API
    for (auto i = 0; i < raw_data_area; ++i) {
        for (auto k = 0; k < c; ++k) {
            cpu_data_ptr[i + k * raw_data_area] =
                (static_cast<float>(raw_image_ptr[i * 3 + k]) / 255.f - mean.at(k)) / std.at(k);
        }
    }

    return fp32_data;
}

template <typename BufferType>
void LoadImageCPUYolox(const std::string& file_name, BufferType* image, const TensorShape& tensor_shape,
                       int32_t batch_idx) {
    if (tensor_shape.data_format != TensorFormat::kHWC) {
        std::cerr << "Image should be load as NHWC format" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (not(batch_idx < tensor_shape.dims.at(0))) {
        std::cerr << "Set image data for batch idx: " << batch_idx << ", out of boundary: " << tensor_shape.dims.at(0)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::shared_ptr<uint8_t> image_ptr{nullptr};
    int32_t w, h, c;
    image_ptr = GetImagePtr(file_name, &w, &h, &c, 0);
    ImageRGB2BGR(w, h, image_ptr.get());
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (tensor_shape.dims.at(2) != h or tensor_shape.dims.at(3) != w) {
        std::cerr << "Image preprocess not include resize, image raw size must "
                     "be same with input setting "
                  << ", image h: " << h << ", w: " << w << " memory h: " << tensor_shape.dims.at(2)
                  << ", w: " << tensor_shape.dims.at(3) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    uint64_t num_element = (tensor_shape.dims.at(1) + tensor_shape.padding.at(1)) *
                           (tensor_shape.dims.at(2) + tensor_shape.padding.at(2)) *
                           (tensor_shape.dims.at(3) + tensor_shape.padding.at(3));

    // Only support NHWC format
    auto padding_value = tensor_shape.padding.back();

    float* cpu_data_ptr = reinterpret_cast<float*>(image->GetDataPtr()) + batch_idx * num_element;
    uint8_t* raw_image_ptr = image_ptr.get();
    // for (auto i = 0; i < 20; i++) {
    //     printf("image [%d] : %d \n", i, raw_image_ptr[i]);
    // }

    // Only set first image content
    uint64_t raw_data_area = w * h;
    uint64_t count{0};
    for (auto i = 0; i < raw_data_area; ++i) {
        for (auto k = 0; k < c; ++k) {
            //  0~255
            *cpu_data_ptr = static_cast<float>(*raw_image_ptr);
            raw_image_ptr++;
            cpu_data_ptr++;
            count++;
        }
        for (auto t = 0; t < padding_value; ++t) {
            *cpu_data_ptr = 0.0f;
            cpu_data_ptr++;
            count++;
        }
    }
    // cpu_data_ptr =
    //     reinterpret_cast<float*>(image->GetDataPtr()) + batch_idx *
    //     num_element;
    // for (auto i = 0; i < 20; i++) {
    //     printf("image cpudataptr [%d] : %f \n", i, cpu_data_ptr[i]);
    // }
    /*
    // Set 0 for the memory range out image
    for (auto i = count; i < input_cpu->GetNumElement(); ++i) {
        *cpu_data_ptr = 0.f;
        cpu_data_ptr++;
    }
    */
}

template <typename BufferType>
void LoadImageCPURetinaFace(const std::string& file_name, BufferType* image, const TensorShape& tensor_shape,
                            float* means, float* stds, int32_t batch_idx) {
    if (tensor_shape.data_format != TensorFormat::kHWC) {
        std::cerr << "Image should be load as NHWC format" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (not(batch_idx < tensor_shape.dims.at(0))) {
        std::cerr << "Set image data for batch idx: " << batch_idx << ", out of boundary: " << tensor_shape.dims.at(0)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::shared_ptr<uint8_t> image_ptr{nullptr};
    int32_t w, h, c;
    image_ptr = GetImagePtr(file_name, &w, &h, &c, 0);
    ImageRGB2BGR(w, h, image_ptr.get());
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (tensor_shape.dims.at(2) != h or tensor_shape.dims.at(3) != w) {
        std::cerr << "Image preprocess not include resize, image raw size must "
                     "be same with input setting "
                  << ", image h: " << h << ", w: " << w << " memory h: " << tensor_shape.dims.at(2)
                  << ", w: " << tensor_shape.dims.at(3) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    uint64_t num_element = (tensor_shape.dims.at(1) + tensor_shape.padding.at(1)) *
                           (tensor_shape.dims.at(2) + tensor_shape.padding.at(2)) *
                           (tensor_shape.dims.at(3) + tensor_shape.padding.at(3));

    // Only support NHWC format
    auto padding_value = tensor_shape.padding.back();

    float* cpu_data_ptr = reinterpret_cast<float*>(image->GetDataPtr()) + batch_idx * num_element;
    uint8_t* raw_image_ptr = image_ptr.get();

    // Only set first image content
    uint64_t raw_data_area = w * h;
    uint64_t count{0};
    for (auto i = 0; i < raw_data_area; ++i) {
        for (auto k = 0; k < c; ++k) {
            //  0~255
            *cpu_data_ptr = (static_cast<float>(*raw_image_ptr) - means[k]) / stds[k];
            raw_image_ptr++;
            cpu_data_ptr++;
            count++;
        }
        for (auto t = 0; t < padding_value; ++t) {
            *cpu_data_ptr = 0.0f;
            cpu_data_ptr++;
            count++;
        }
    }
}

template void LoadImageCPU(const std::string& file_name, HostBuffer* image, const std::vector<int32_t>& tensor_shape,
                           int32_t batch_idx, bool hwc, const std::vector<float>& mean, const std::vector<float>& std,
                           bool convert_to_bgr);
template void LoadImageCPU(const std::string& file_name, HostPinnedBuffer* image,
                           const std::vector<int32_t>& tensor_shape, int32_t batch_idx, bool hwc,
                           const std::vector<float>& mean, const std::vector<float>& std, bool convert_to_bgr);
template void LoadImageCPUYolox(const std::string& file_name, HostBuffer* image, const TensorShape& tensor_shape,
                                int32_t batch_idx);
template void LoadImageCPUYolox(const std::string& file_name, HostPinnedBuffer* image, const TensorShape& tensor_shape,
                                int32_t batch_idx);
template void LoadImageCPURetinaFace(const std::string& file_name, HostBuffer* image, const TensorShape& tensor_shape,
                                     float* means, float* stds, int32_t batch_idx);
template void LoadImageCPURetinaFace(const std::string& file_name, HostPinnedBuffer* image,
                                     const TensorShape& tensor_shape, float* means, float* stds, int32_t batch_idx);
void LoadImageBuffer(const std::string& file_name, const nvinfer1::Dims& dims, float* data, int32_t batch) {
    int32_t iw, ih, c;
    auto image_ptr = GetImagePtr(file_name, &iw, &ih, &c, 0);
    if (not image_ptr) {
        std::cerr << "File " << file_name << " does not exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (not(c == 3 or c == 1)) {
        std::cout << "Image channel is " << c << std::endl;
    }
    assert(dims.nbDims == 4);
    int ow = dims.d[3];
    int oh = dims.d[2];
    auto* odata = (unsigned char*)malloc(ow * oh * c);
    stbir_resize(image_ptr.get(), iw, ih, 0, odata, ow, oh, 0, STBIR_TYPE_UINT8, c, STBIR_ALPHA_CHANNEL_NONE, 0,
                 STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_BOX, STBIR_FILTER_BOX, STBIR_COLORSPACE_SRGB,
                 nullptr);

    uint64_t num_element = ow * oh * c;
    if (batch < 0) {
        batch = 0;
    }

    float* cpu_data_ptr = data + num_element * batch;
    uint8_t* raw_image_ptr = odata;
    // Only set first image content
    uint64_t raw_data_area = ow * oh;
    // From HWC to CHW
    for (auto i = 0; i < raw_data_area; ++i) {
        for (auto k = 0; k < c; ++k) {
            cpu_data_ptr[i + k * raw_data_area] = static_cast<float>(raw_image_ptr[i * c + k]) / 255.f;
        }
    }
    stbi_image_free(odata);
}
