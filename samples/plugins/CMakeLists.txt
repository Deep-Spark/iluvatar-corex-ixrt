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

cmake_minimum_required(VERSION 3.10)

project(oss_ixrt_plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(TARGET_NAME oss_ixrt_plugin)
set(SHARED_TARGET ${TARGET_NAME})

set(PLUGIN_REPO_PATH ${PROJECT_SOURCE_DIR})
set(PLUGIN_SOURCES)

include(FindCompiler)
include(FindCuda)
file(GLOB_RECURSE PLUGIN_FILES ${PLUGIN_REPO_PATH}/src/*.cpp
        ${PLUGIN_REPO_PATH}/src/*.cc
        ${PLUGIN_REPO_PATH}/src/*.cu)
list(APPEND PLUGIN_FILES ${PLUGIN_SOURCES})
include_directories(${IXRT_INCLUDE_DIR}
        ${CUDA_PATH}/include
        /opt/sw_home/local/cuda/include)
################################## Compile Options ######################################
# For cuda files
string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_PATH}")

cuda_add_library(${SHARED_TARGET} SHARED
        ${PLUGIN_FILES}
)
target_link_libraries(${SHARED_TARGET} PUBLIC  cublasLt cudart ixrt)
target_link_directories(${SHARED_TARGET} PUBLIC ${IXRT_LIB_DIR})
target_include_directories(${SHARED_TARGET} PUBLIC src PUBLIC src/common PUBLIC  src/common/kernels)
set_target_properties(${SHARED_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
