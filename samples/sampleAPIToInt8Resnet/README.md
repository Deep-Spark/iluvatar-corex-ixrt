# Building Resnet18 Layer By Layer In IxRT

This sample, sampleAPIToInt8Resnet, uses the API to build int8 Resnet18 layer by layer, sets up weights and inputs/outputs and then performs inference.

## Setup

Please refer to official documentation to install IxRT library.

## Gen Model Weight File

```bash
cd samples/sampleAPIToInt8Resnet
python3 quant_int_wts.py --imagenet_val /path/to/imagenet/val
```

## Running the sample

### Compile the samples in IxRT OSS directory

If you haven't compiled the IxRT samples, following command helps you compile all samples.
Refer to the README section of ixrt-oss.

```
cd ixrt-oss
cmake -B build -DIXRT_HOME=ixrt-oss/IxRT
cmake --build build -j
```

### Run

```bash
./build/bin/sampleAPIToInt8Resnet
```
