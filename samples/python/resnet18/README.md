All old api examples please refer to [run.sh](./run.sh)

# ResNet trt api Inference Sample
## Prequisities

### Test data
Download the sample data from the IxRT release tarball.

### Install IxRT library
Follow steps from IxRT's official document.

## Running the sample

### Run in fixed shape

#### Test int8
```
python3 resnet18.py --model_path ../../../data/resnet18_ckpt/
```

#### Test fp16
```
python3 resnet18.py --model_path ../../../data/resnet18_ckpt/ --precision float16
```

### Run in multi contexts
```shell
python3 resnet18.py --model_path ../../../data/resnet18_ckpt/ --multicontext
```

### Run with dynamic shape
```shell
python3 resnet18.py --model_path ../../../data/resnet18_ckpt/ --dynamicshape
```

### Run with dynamic shape in multi contexts
```shell
python3 resnet18.py --model_path ../../../data/resnet18_ckpt/ --dynamicshape_multicontext
```

### Register runtime hook
```bash
ixrtexec --onnx data/resnet18/resnet18.onnx --save_engine /tmp/resnet18.engine
python3 sample_hook.py --engine_path /tmp/resnet18.engine
```

## Parameters
|name | |
|--|--|
| --model_path| folder contains the onnx and image|
| --use_async| use async excute methed|
| --precision| model execute precision, "float16" or "int8"|
| --multicontext| use multi contexts in running test|
| --dynamicshape| use dynamic input shapes in running test|
| --dynamicshape_multicontext| use dynamic input shapes and multi contexts in running test|
