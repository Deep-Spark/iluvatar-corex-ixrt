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

# ResNet18
SCRIPTPATH=$(dirname $(realpath "$0"))
DATA=${SCRIPTPATH}/../../../data/resnet18/
## Example1: Infer with Int8 with CPU IO

# python3 0.quant.py
# python3 1.build_engine.py
# python3 2.load_engine.py

## Example2: Int8 infer with GPUIO
# python3 3.load_engine_and_use_gpu_io.py

## Example3: Float16 infer with CPUIO
python3 1.build_engine.py --onnx-path ${DATA}/resnet18.onnx --precision float16
python3 2.load_engine.py
