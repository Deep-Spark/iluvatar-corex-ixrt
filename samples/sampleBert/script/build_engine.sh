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

BSZ=1
USE_FP16=True

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --int8) USE_FP16=False;;
    esac
done

if [ "$USE_FP16" = "True" ]; then
    echo 'USE_FP16=True'
    python3 builder.py -x ./data/bert-large-uncased/bert_large_v1_1_fake_quant.onnx \
                       -w 4096 \
                       -o ./data/bert_large_384.engine \
                       -s 1 384 384 \
                       -b 1 ${BSZ} ${BSZ} \
                       --fp16 \
                       -c ./data/bert-large-uncased/bert_config.json
else
    echo 'USE_INT8=True'
    python3 builder_int8.py -pt ./data/bert-large-uncased/bert_large_int8_qat.bin \
                -o ./data/bert_large_384_int8.engine \
                -s 1 384 384 \
                -b 1 ${BSZ} ${BSZ} \
                -i \
                -c ./data/bert-large-uncased/bert_config.json
fi
