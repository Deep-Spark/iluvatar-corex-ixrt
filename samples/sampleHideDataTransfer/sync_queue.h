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

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

template <typename T>
class SyncQueue {
   public:
    SyncQueue(int32_t max_size) : max_size_(max_size), working_(true) {}
    bool Push(const T& x) {
        std::unique_lock<std::mutex> locker(mutex_);
        full_cv_.wait(locker, [this] { return (not IsFull()) and working_; });
        if (not working_) {
            return false;
        }
        queue_.push(x);
        empty_cv_.notify_one();
        return true;
    }

    bool Take(T& x) {
        std::unique_lock<std::mutex> locker(mutex_);
        empty_cv_.wait(locker, [this] { return (not IsEmpty()) and working_; });
        if (not working_) {
            return false;
        }
        x = queue_.front();
        queue_.pop();
        full_cv_.notify_one();
        return true;
    }

    void Clear() {
        working_ = false;
        empty_cv_.notify_one();
        full_cv_.notify_one();
        while (not queue_.empty()) {
            queue_.pop();
        }
        working_ = true;
    }

    uint64_t Size() {
        std::unique_lock<std::mutex> locker(mutex_);
        return queue_.size();
    }

    bool Full() {
        std::unique_lock<std::mutex> locker(mutex_);
        return IsFull();
    }

    bool Empty() {
        std::unique_lock<std::mutex> locker(mutex_);
        return IsEmpty();
    }

   private:
    bool IsFull() const {
        if (max_size_ <= 0) {
            return false;
        } else {
            return queue_.size() == max_size_;
        }
    }

    bool IsEmpty() const { return queue_.empty(); }

    int32_t max_size_;
    std::queue<T> queue_;
    std::condition_variable full_cv_;
    std::condition_variable empty_cv_;
    std::mutex mutex_;
    bool working_;
};
