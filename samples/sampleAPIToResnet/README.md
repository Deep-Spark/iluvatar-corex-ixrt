# Building Resnet18 Layer By Layer In IxRT

This sample, sampleAPIToResnet, uses the API to build Resnet18 layer by layer, sets up weights and inputs/outputs and then performs inference.

## Setup

Please refer to official documentation to install IxRT library.

## Gen Model Weight File

```bash
cd samples/sampleAPIToResnet
python3 gen_wts.py
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
./build/bin/sampleAPIToResnet
```
