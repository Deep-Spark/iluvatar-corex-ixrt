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
#include <cassert>
#include <iostream>

#include "NvInfer.h"
using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger {
   public:
    explicit Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}

    //!
    //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
    //! \return The nvinfer1::ILogger associated with this Logger
    //!
    //! TODO Once all samples are updated to use this method to register the logger with IxRT,
    //! we can eliminate the inheritance of Logger from ILogger
    //!
    nvinfer1::ILogger& getIxRTLogger() noexcept { return *this; }

    //!
    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    //!
    //! Note samples should not be calling this function directly; it will eventually go away once we eliminate the
    //! inheritance from nvinfer1::ILogger
    //!
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= mReportableSeverity) {
            std::cout << severityPrefix(mReportableSeverity) << "[IXRT] " << msg << std::endl;
        }
    }

    //!
    //! \brief Method for controlling the verbosity of logging output
    //!
    //! \param severity The logger will only emit messages that have severity of this level or higher.
    //!
    void setReportableSeverity(Severity severity) noexcept { mReportableSeverity = severity; }

    //!
    //! \brief Opaque handle that holds logging information for a particular test
    //!
    //! This object is an opaque handle to information used by the Logger to print test results.
    //! The sample must call Logger::defineTest() in order to obtain a TestAtom that can be used
    //! with Logger::reportTest{Start,End}().
    //!

    Severity getReportableSeverity() const { return mReportableSeverity; }

   private:
    //!
    //! \brief returns an appropriate string for prefixing a log message with the given severity
    //!
    static const char* severityPrefix(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                return "[F] ";
            case Severity::kERROR:
                return "[E] ";
            case Severity::kWARNING:
                return "[W] ";
            case Severity::kINFO:
                return "[I] ";
            case Severity::kVERBOSE:
                return "[V] ";
            default:
                assert(0);
                return "";
        }
    }

    Severity mReportableSeverity;
};  // class Logger
