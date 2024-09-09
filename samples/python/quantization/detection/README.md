# YoloV5m PTQ Guidance
## Prequisities

### Test data
Download the sample data from the IxRT release tarball.

### Install IxRT library
Follow steps from IxRT's official document.

## Running the sample
``` bash
export DATA_PATH=/home/datasets/cv/coco2017

python3 ptq_detection.py \
    --model yolov5m_without_decoder.onnx \
    --batch_size 16 \
    --img_size 640  \
    --data_path ${DATA_PATH}   \
    --num_samples 400 \
    --disable_bias_correction  \
    --fp32_acc
```
Arguments:
- `--model`, the input yolov5 onnx without decoders.
- `--batch_size`, batch size of the input.
- `--img_size`, H or W of the model.
- `--data_path`, the dataset path for COCO2017.
- `--num_samples`, number of images as calibration dataset.
- `--disable_bias_correction`, disable bias correction, so the bias in the model will be unchanged.
- `--fp32_acc`, whether to evaluate the original model.
