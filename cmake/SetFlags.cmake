# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

option(USE_TRT "Use another inference framework for API comparison" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)
if (USE_TRT)
    set(LIBRT nvinfer)
    set(LIBPLUGIN nvinfer_plugin)
    set(LIBPARSER nvonnxparser)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_TRT")
else()
    set(LIBRT ixrt)
    set(LIBPLUGIN ixrt_plugin)
    set(LIBPARSER "")
endif ()
