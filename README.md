# IxRT Open Source Software

This repository contains the Open Source Software (OSS) components of the Iluvatar Corex IxRT. It includes the sources for IxRT plugins and deploy tools, as well as sample applications demonstrating the usages and capabilities of the IxRT platform.

Iluvatar CoreX IxRT is a high-performance AI inference engine. IxRT provides an AI compiler, an inference runtime and development APIs for AI inference applications.

Directory structure:

```bash
.
├── cmake # Various cmake scripts
├── ixrt # IxRT Python code, include its deploy tool
├── requirements # Python requirements
├── samples # IxRT samples
├── tools # IxRT various tools, please check readme for details
├── CMakeLists.txt # To build samples
├── README.md # This readme
```

## Prebuilt IxRT Python Package

We provide the IxRT Python package for an easy installation.
To install:

```bash
pip3 install ixrt
```

## Build

### Prerequisites

To build the IxRT-OSS components, you will first need some software packages, The required dependencies for installation are located in the folder named requirements.

```bash
cd $IxRT_OSS_PATH/requirements
pip3 install -r requirements-ixrt.txt
pip3 install -r requirements-quant-tests.txt
pip3 install -r requirements-quant.txt
```

### Building IxRT-OSS

#### Generate Makefiles and build

To build IxRT C++ samples, you will also need to download IxRT `.run` or `.tar.gz` software package from Iluvatar official website, and install it following procedures in **Install** chapter of IxRT documentation.

```bash
cd ixrt-oss
wget ixrt-<version>+corex.x.x.x-linux_x86_64.tar.gz
tar -xzvf ixrt-<version>+corex.x.x.x-linux_x86_64.tar.gz
cmake -B build -DIXRT_HOME=ixrt-oss/IxRT
cmake --build build -j
```

## Samples Lists

Below is a sample of the model running with IxRT.

| Sample                                      | Language | Format | Description                                                |
| ------------------------------------------- | -------- | ------ | ---------------------------------------------------------- |
| [sampleYoloV3](samples/sampleYoloV3)        | C++      | ONNX   | Execute YoloV3 with IxRT API                               |
| [sampleYoloX](samples/sampleYoloX)          | C++      | ONNX   | Execute YoloX-m with IxRT API                              |
| [sampleYoloV5](samples/sampleYoloV5)        | C++      | ONNX   | Execute YoloV5-s with IxRT API                             |
| [resnet18](samples/python/resnet18)         | Python   | ONNX   | Execute ResNet18 with IxRT API                             |
| [sampleResNet](samples/sampleResNet)        | C++      | ONNX   | Execute ResNet18 with IxRT API                             |
| [quantization](samples/python/quantization) | Python   | ONNX   | Demonstrate how to useIxRT deploy tool to quantize an onnx |

### Advanced samples

| Sample                                                   | Language | Format | Description                                                                              |
| -------------------------------------------------------- | -------- | ------ | ---------------------------------------------------------------------------------------- |
| [sampleHideDataTransfer](samples/sampleHideDataTransfer) | C++      | ONNX   | Hide data transfer with ResNet18 ONNX                                                    |
| [sampleBert](samples/sampleBert)                         | Python   | ONNX   | Construct an int8 and fp16 Bert with IxRT custom layer API                               |
| [sampleQuant](samples/sampleQuant)                       | C++      | ONNX   | Demonstrate how to useIxRT deploy tool to quantize an onnx                               |
| [python](samples/python)                                 | Python   | ONNX   | IXRT Quantization and Example Tool Using Python API                                      |
| [plugins](samples/plugins)                               | C++      | ONNX   | The plugin library provided by IxRT, which includes the NMSplugin and YOLOdecoder plugin |

## License

Copyright (c) 2024 Iluvatar CoreX. All rights reserved. This project has an Apache-2.0 license, as found in the [LICENSE](LICENSE) file.
