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
#include <algorithm>
#include <fstream>
#include <vector>

#include "NvInfer.h"
#include "misc.h"

namespace SampleHelper {
class IBatchStream {
   public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class BatchStream : public IBatchStream {
   public:
    BatchStream(int batchSize, int maxBatches, int h, int w, int nStep, const std::vector<std::string>& directories);

    // Resets data members
    void reset(int firstBatch) override;

    bool next() override;
    void skip(int skipCount) override;

    float* getBatch() override { return mBatch.data(); }

    int getBatchesRead() const override { return mBatchCount; }

    int getBatchSize() const override { return mBatchSize; }

    nvinfer1::Dims getDims() const override { return mDims; }

   private:
    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    std::vector<float> mBatch;           //!< Data for the batch
    std::vector<float> mFileBatch;       //!< List of image files
    nvinfer1::Dims mDims;                //!< Input dimensions
    std::vector<std::string> mDataDirs;  //!< Directories where the files can be found
    std::vector<std::string> mFilePaths;
};
}  // end of namespace SampleHelper
