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

# This cmake does:
# - Set CUDA_PATH
# - Find libcudart
# - Util functions like cuda_add_library, cuda_add_executable


# CUDA_PATH can be specified through below means shown in priority order 1.
# cmake command line argument, -DCUDA_PATH=/path/to/cuda 2. bash environment
# variable, export CUDA_PATH=/path/to/cuda
if(DEFINED ENV{CUDA_PATH})
  set(CUDA_PATH "$ENV{CUDA_PATH}")
else()
  set(CUDA_PATH
      "/usr/local/corex"
      CACHE PATH "cuda installation root path")
endif()
message(STATUS "Use CUDA_PATH=${CUDA_PATH} ")
link_directories(${CUDA_PATH}/lib)

macro(cuda_add_executable)
  foreach(File ${ARGN})
    if(${File} MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
  add_executable(${ARGV})
endmacro()

macro(cuda_add_library)
  foreach(File ${ARGN})
    if(${File} MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
  add_library(${ARGV})
endmacro()

find_library(
  CUDART_LIBRARY cudart
  PATHS ${CUDA_PATH}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH)

message(STATUS "CUDART_LIBRARY: ${CUDART_LIBRARY}")
if (NOT USE_TRT)
  set(CUDA_LIBRARIES cudart)
endif()
