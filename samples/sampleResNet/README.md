# ResNet INT8 Inference Sample
## Prequisities

### Test data
Download the sample data from the IxRT release tarball.

### Install IxRT library
Follow steps from IxRT's official document.

## Running the sample
0. Quantization
With the help of IxRT deploy tool, quantize it to INT8 QDQ model
```bash
DATADIR=$(realpath data/resnet18)
python3 samples/sampleResNet/quant.py --model ${DATADIR}/resnet18.onnx --save_model_path ${DATADIR}/resnet18_qdq.onnx
```

1. Compile the samples in IxRT OSS directory

If you haven't compiled the IxRT samples, following command helps you compile all samples.
Refer to the README section of ixrt-oss.

```
cd ixrt-oss
cmake -B build -DIXRT_HOME=ixrt-oss/IxRT
cmake --build build -j
```

2. Run
```bash
./sampleResNet [option]

# examples:
./sampleResNet tex_i8
./sampleResNet tex_fp16
```

3. About demo options

| option | description                                                           |
| ------ |-----------------------------------------------------------------------|
| tex_i8_explicit | run resnet18 int8 demo with Q/DQ node in onnx file                    |
| tex_i8_implicit | run resnet18 int8 demo with normal onnx file and tensor dynamic range |
| tex_fp16 | run resnet18 fp16 demo                                                |
| tex_fp32 | run resnet18 fp32 demo                                                |
| tex_s_onnx | run resnet18 from separated ONNX file                                 |
| ten | run resnet18 with enqueueV2 method                                    |
| tmc | run resent18 with multi execution context                             |
| ted | run resnet18 with dynamic shape                                       |
| tmcd | run resnet18 with multi execution context and dynamic shape           |
| load_engine | run resnet18 without building engine procedure                        |
| hook | run resnet18 with hook to acquire tensor data                         |
