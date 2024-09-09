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

source ${1}

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bsz) BSZ=${arguments[index]};;
      --precision) PRECISION=${arguments[index]};;
    esac
done

step=0
SIM_MODEL=${YOLOV5_CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
echo YOLOV5_CHECKPOINTS_DIR : ${YOLOV5_CHECKPOINTS_DIR}
if [ ! -d $YOLOV5_CHECKPOINTS_DIR ];then
    mkdir -p $YOLOV5_CHECKPOINTS_DIR
fi

# Simplify Model
let step++
echo [STEP ${step}] : Simplify Model
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model, ${SIM_MODEL} has been existed
else
    # cp ${ORIGINE_MODEL} ${SIM_MODEL}
    python3 ./samples/sampleYoloV5/deploy/simplify_model.py \
    --origin_model $ORIGINE_MODEL    \
    --output_model ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi

# Quant Model
if [ $PRECISION == "int8" ];then
    let step++
    echo;
    echo [STEP ${step}] : Quant Model
    if [[ -z ${QUANT_EXIST_ONNX} ]];then
        QUANT_EXIST_ONNX=$YOLOV5_CHECKPOINTS_DIR/quantized_${MODEL_NAME}.onnx
    fi
    if [ -z ${QUANT_EXIST_SCALE_JSON} ];then
        QUANT_EXIST_SCALE_JSON=$YOLOV5_CHECKPOINTS_DIR/quantized_${MODEL_NAME}.json
    fi
    if [[ -f ${QUANT_EXIST_ONNX} && -f ${QUANT_EXIST_SCALE_JSON} ]];then
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Quant Model Skip, ${QUANT_EXIST_ONNX} and ${QUANT_EXIST_SCALE_JSON} has been existed
    else
        python3 ./samples/sampleYoloV5/deploy/quant.py              \
            --model ${SIM_MODEL}               \
            --model_name ${MODEL_NAME}         \
            --dataset_dir ${DATASET_DIR}       \
            --observer ${QUANT_OBSERVER}       \
            --disable_quant_names ${DISABLE_QUANT_LIST[@]} \
            --save_dir $YOLOV5_CHECKPOINTS_DIR        \
            --bsz   ${QUANT_BATCHSIZE}         \
            --step  ${QUANT_STEP}              \
            --seed  ${QUANT_SEED}              \
            --imgsz ${IMGSIZE}                 \
            --quant_format ${QUANT_FORMAT}
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Generate ${SIM_MODEL} and ${QUANT_EXIST_SCALE_JSON}
    fi
else
    QUANT_EXIST_SCALE_JSON=tmp
fi

# Fuse Layer
if [ $LAYER_FUSION == "1" ]; then
    let step++
    echo;
    echo [STEP ${step}] : Layer Fusion
    FINAL_ONNX=${YOLOV5_CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_fusion.onnx
    if [ -f $FINAL_ONNX ];then
        echo "  "Layer Fusion Skip, $FINAL_ONNX has been existed
    else
        python3 ./samples/sampleYoloV5/deploy/fuse_model.py        \
            --origin_model $SIM_MODEL       \
            --output_model ${FINAL_ONNX}    \
            --bsz   ${BSZ}                  \
            --imgsz ${IMGSIZE}              \
            --faster_impl ${DECODER_FASTER}
    fi
else
    FINAL_ONNX=$SIM_MODEL
fi

mv $FINAL_ONNX ${YOLOV5_CHECKPOINTS_DIR}/yolov5_final.onnx
echo "Final onnx : " ${YOLOV5_CHECKPOINTS_DIR}/yolov5_final.onnx
