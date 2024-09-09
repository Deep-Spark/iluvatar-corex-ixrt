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
If you haven't compiled the IxRT samples, following command helps you compile all samples
```
cd path-to-ixrt-oss
cmake -B build
cmake --build build -j
```

2. Run
```bash
./sampleResNet [option]

# examples:
./sampleResNet tex_i8
./sampleResNet tex_fp16
```
