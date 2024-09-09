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

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND cat /etc/os-release OUTPUT_VARIABLE ETC_OS_RELEASE)
    if (ETC_OS_RELEASE MATCHES "Ubuntu")
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()
set(OPENSOURCE_SAMPLES_LIST
     sampleYoloV5
     sampleResNet
     sampleYoloX
     sampleHideDataTransfer
     sampleAPIToResnet
     sampleYoloV3
     sampleAPIToInt8Resnet
     sampleQuant
)

add_subdirectory(common)
add_subdirectory(plugins)
foreach(SAMPLE_ITER ${OPENSOURCE_SAMPLES_LIST})
    add_subdirectory(${SAMPLE_ITER})
endforeach(SAMPLE_ITER)
