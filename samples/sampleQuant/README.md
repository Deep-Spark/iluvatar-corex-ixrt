# Implicit quantization In IxRT

This sample, sampleQuant, to demonstrate the usage of implicit quantization in IxRT.

## Setup

Please refer to official documentation to install IxRT library.

## Download the quantization dataset.

```bash
cd path-to-sampleQuant
bash prepare.sh
```

## Running the sample

### Compile the samples in IxRT OSS directory

If you haven't compiled the IxRT samples, following command helps you compile all samples.

```
cd path-to-ixrt-oss
cmake -B build
cmake --build build -j
```

### Run

```bash
./build/bin/sampleQuant tex_quant
```
