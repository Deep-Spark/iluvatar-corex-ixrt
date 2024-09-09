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

#include <cstring>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "NvInfer.h"
#include "batch_stream.h"

using namespace SampleHelper;

namespace nvinfer1 {

namespace entropy_calib_helper {
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)
#endif

#undef CHECK_TRUE
#define CHECK_TRUE(value, commit)                                                                             \
    {                                                                                                         \
        if (not(value)) {                                                                                     \
            std::cerr << __FILE__ << ":" << __LINE__ << " -" << __FUNCTION__ << " : " << commit << std::endl; \
            std::exit(EXIT_FAILURE);                                                                          \
        }                                                                                                     \
    }

}  // end of namespace entropy_calib_helper

using namespace entropy_calib_helper;

template <typename TBatchStream>
class EntropyCalibratorImpl {
   public:
    EntropyCalibratorImpl(TBatchStream stream, int firstBatch, std::string networkName,
                          std::vector<std::string> inputBlobNames, bool readCache = true);
    virtual ~EntropyCalibratorImpl();
    int getBatchSize() const noexcept;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept;
    bool getBatch(void** bindings, const char* names, int nbBindings) noexcept;
    const void* readCalibrationCache(size_t& length) noexcept;
    void writeCalibrationCache(const void* cache, size_t length) noexcept;

   private:
    TBatchStream stream_;
    bool bReadCache_{true};

    std::string calib_table_name_;

    size_t nb_inputs_;
    std::vector<std::string> input_names_;
    std::vector<void*> device_inputs_;
    std::vector<nvinfer1::Dims> dims_list_;

    std::vector<char> calib_cache_;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
   public:
    Int8EntropyCalibrator2(TBatchStream stream, int firstBatch, const std::string& networkName,
                           const std::vector<std::string>& inputBlobNames, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobNames, readCache) {}

    int getBatchSize() const noexcept override { return mImpl.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    bool getBatch(void** bindings, const char* names, int nbBindings) noexcept {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override { return mImpl.readCalibrationCache(length); }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        mImpl.writeCalibrationCache(cache, length);
    }

   private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

template <typename TBatchStream>
EntropyCalibratorImpl<TBatchStream>::EntropyCalibratorImpl(TBatchStream stream, int firstBatch, std::string networkName,
                                                           std::vector<std::string> inputBlobNames, bool readCache)
    : stream_{stream}, calib_table_name_("CalibrationTable" + networkName), bReadCache_(readCache) {
    input_names_.assign(inputBlobNames.begin(), inputBlobNames.end());

    nb_inputs_ = inputBlobNames.size();

    dims_list_ = {stream_.getDims()};
    device_inputs_.resize(nb_inputs_, nullptr);
    for (int i = 0; i < nb_inputs_; ++i) {
        const auto& d = dims_list_[i];
        auto vol = std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        CUDA_CHECK(cudaMalloc((void**)(&device_inputs_[i]), vol * sizeof(float)));
    }

    stream_.reset(firstBatch);
}

template <typename TBatchStream>
EntropyCalibratorImpl<TBatchStream>::~EntropyCalibratorImpl() {
    for (int i = 0; i < nb_inputs_; ++i) {
        CUDA_CHECK(cudaFree(device_inputs_[i]));
    }
}

template <typename TBatchStream>
int EntropyCalibratorImpl<TBatchStream>::getBatchSize() const noexcept {
    return stream_.getBatchSize();
}

template <typename TBatchStream>
bool EntropyCalibratorImpl<TBatchStream>::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    CHECK_TRUE(nbBindings <= nb_inputs_, "nbBindings must be less than number of inputs.");

    if (!stream_.next()) {
        return false;
    }

    for (int i = 0; i < nbBindings; ++i) {
        CHECK_TRUE(!strcmp(names[i], input_names_[i].c_str()), "Name not match in getBatch.");

        const auto& d = dims_list_[i];
        auto vol = std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        CUDA_CHECK(cudaMemcpy(device_inputs_[i], stream_.getBatch(), vol * sizeof(float), cudaMemcpyHostToDevice));

        bindings[i] = (void*)device_inputs_[i];
    }

    return true;
}

template <typename TBatchStream>
bool EntropyCalibratorImpl<TBatchStream>::getBatch(void** bindings, const char* names, int nbBindings) noexcept {
    if (!stream_.next()) {
        return false;
    }

    const auto& d = dims_list_[0];
    auto vol = std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
    CUDA_CHECK(cudaMemcpy(device_inputs_[0], stream_.getBatch(), vol * sizeof(float), cudaMemcpyHostToDevice));

    *bindings = (void*)device_inputs_[0];
    return true;
}

template <typename TBatchStream>
const void* EntropyCalibratorImpl<TBatchStream>::readCalibrationCache(size_t& length) noexcept {
    calib_cache_.clear();
    std::cout << "Load calibration cache from " << calib_table_name_ << std::endl;
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (bReadCache_ && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

template <typename TBatchStream>
void EntropyCalibratorImpl<TBatchStream>::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::cout << "Write calibration cache to " << calib_table_name_ << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

}  // namespace nvinfer1
