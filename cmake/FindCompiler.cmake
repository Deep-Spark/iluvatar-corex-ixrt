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

if(NOT COMPILER_PATH)
  if (EXISTS /opt/sw_home/local/bin/clang++)
    set(COMPILER_PATH /opt/sw_home/local/bin)
  elseif (EXISTS /usr/local/corex/bin/clang++)
    set(COMPILER_PATH /usr/local/corex/bin)
  else()
    message(STATUS "COMPILER_PATH is not set and we couldn't find clang compiler neither, will use system C/C++ compiler")
  endif()
endif()
if (COMPILER_PATH)
set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++)
set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang)
endif()
message(STATUS "Use ${CMAKE_CXX_COMPILER} and ${CMAKE_C_COMPILER} as C++ and C compiler")
