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
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(TARGET_NAME oss_ixrt_plugin)
set(SHARED_TARGET ${TARGET_NAME})

set(PLUGIN_REPO_PATH ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE PLUGIN_FILES ${PLUGIN_REPO_PATH}/src/**/*.cpp
        ${PLUGIN_REPO_PATH}/src/**/*.cc
        ${PLUGIN_REPO_PATH}/src/**/*.cu)
list(APPEND PLUGIN_FILES ${PLUGIN_SOURCES})
include_directories(${IXRT_INCLUDE_DIR}
        ${CUDA_PATH}/include
        /opt/sw_home/local/cuda/include)
cuda_add_library(${SHARED_TARGET} SHARED
        ${PLUGIN_FILES}
)
target_link_libraries(${SHARED_TARGET} PUBLIC  cublasLt cudart ixrt)
if(INJECT_COMMIT_ID)
add_custom_command(
    TARGET ${SHARED_TARGET}
    POST_BUILD
    COMMAND ${PROJECT_SOURCE_DIR}/tools/dist/commit_cli.sh -f ../../../lib/liboss_ixrt_plugin.so
)
endif()
target_link_directories(${SHARED_TARGET} PUBLIC ${IXRT_LIB_DIR})
target_include_directories(${SHARED_TARGET} PUBLIC src PUBLIC src/common PUBLIC  src/common/kernels)
set_target_properties(${SHARED_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
