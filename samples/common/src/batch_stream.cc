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
#include "batch_stream.h"

#include <dirent.h>
#include <sys/types.h>

#include <fstream>
#include <stack>
#include <string>

#include "image_io.h"

namespace nvinfer1::samples::common {

static bool LoadFilePathsStack(std::string root_path, std::vector<std::string>& file_paths, bool subContain = true) {
    DIR* pDir = nullptr;
    struct dirent* ptr = nullptr;
    if (!(pDir = opendir(root_path.c_str()))) {
        return false;
    }

    auto ComposeFullFile = [](std::string root_path, std::string file) {
        if (not(root_path[root_path.size() - 1] == '/')) {
            return root_path + "/" + file;
        } else {
            return root_path + file;
        }
    };

    file_paths.clear();
    std::stack<std::string> visited_paths;
    visited_paths.emplace(root_path);

    while (not visited_paths.empty()) {
        std::string cur_path = visited_paths.top();
        visited_paths.pop();

        pDir = opendir(cur_path.c_str());
        if (not pDir) {
            continue;
        }

        while ((ptr = readdir(pDir)) != 0) {
            std::string dir_name = ptr->d_name;
            if (dir_name == std::string(".") or dir_name == std::string("..")) {
                continue;
            }

            std::string new_path = ComposeFullFile(cur_path, ptr->d_name);
            if (ptr->d_type == DT_DIR and (subContain)) {
                visited_paths.emplace(new_path);
            } else if (ptr->d_type == DT_REG) {
                file_paths.emplace_back(new_path);
            } else {
                continue;
            }
        }
    }

    closedir(pDir);
    return true;
}

BatchStream::BatchStream(int batchSize, int maxBatches, int h, int w, int nStep,
                         const std::vector<std::string>& directories)
    : mBatchSize(batchSize), mMaxBatches(maxBatches), mDataDirs(directories) {
    mDataDirs.assign(directories.begin(), directories.end());
    if (mDataDirs.empty()) {
        return;
    }

    mDims.nbDims = 4;
    mDims.d[0] = mBatchSize;
    mDims.d[1] = 3;
    mDims.d[2] = h;
    mDims.d[3] = w;

    int nlen = volume(mDims);
    mBatch.resize(nlen, 0.0f);

    for (const auto& dir_name : mDataDirs) {
        std::vector<std::string> file_paths;
        if (not LoadFilePathsStack(dir_name, file_paths, true)) {
            continue;
        }
        for (const auto& full_path : file_paths) {
            auto pos = full_path.find_last_of(".");
            auto postfix = full_path.substr(pos + 1, std::string::npos);
            if (not(postfix == "JPEG" or postfix == "jpg")) {
                continue;
            } else {
                mFilePaths.emplace_back(full_path);
            }
        }
    }

    size_t total_need = nStep * mBatchSize;  // Drop last.
    auto nonrep_num = mFilePaths.size();

    size_t current_pos{0};
    while (mFilePaths.size() < total_need) {
        auto idx = current_pos % nonrep_num;
        mFilePaths.emplace_back(mFilePaths.at(idx));
        current_pos++;
    }

    reset(0);
}

void BatchStream::reset(int firstBatch) {
    mBatchCount = 0;
    skip(firstBatch);
}

// Advance to next batch and return true, or return false if there is no batch left.
bool BatchStream::next() {
    if (mBatchCount > mMaxBatches - 1) {
        return false;
    }
    auto current_pos = mBatchCount * mBatchSize;
    for (auto idx = 0; idx < mBatchSize; ++idx) {
        const auto& file_path = mFilePaths.at(current_pos);
        std::cout << "The " << current_pos << " calibration file: " << file_path << std::endl;
        LoadImageBuffer(file_path, mDims, mBatch.data(), idx);
        current_pos++;
    }
    mBatchCount++;
    return true;
}

// Skips the batches
void BatchStream::skip(int skipCount) {
    for (int i = 0; i < skipCount; i++) {
        next();
    }
    mBatchCount = skipCount;
}

}  // end of namespace nvinfer1::samples::common