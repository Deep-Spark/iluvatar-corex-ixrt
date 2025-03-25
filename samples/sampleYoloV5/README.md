# YoloV5
## Prequisities

### Prepare data and checkpoints

```
export YOLOV5_CHECKPOINTS_DIR=./data/yolov5/
```

#### Generate float16 fusion onnx
```
bash samples/sampleYoloV5/deploy.sh samples/sampleYoloV5/config/YOLOV5S_CONFIG --precision float16
```

### Install IxRT library
Follow steps from IxRT's official document.

## Running the sample
1. Compile the samples in IxRT OSS directory

If you haven't compiled the IxRT samples, following command helps you compile all samples.
Refer to the README section of ixrt-oss.

```
cd ixrt-oss
cmake -B build -DIXRT_HOME=ixrt-oss/IxRT
cmake --build build -j
```

2. Run

run fp16
```bash
./build/bin/sampleYoloV5 --precision fp16 --data_dir $YOLOV5_CHECKPOINTS_DIR
```
