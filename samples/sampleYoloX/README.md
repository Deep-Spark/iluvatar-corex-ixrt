# YOLOX-m in IxRT

Support for YOLOX-m model in IxRT. This script helps with converting and running this model with IxRT.

## Setup

Install IxRT as per the [IxRT install](ixrt/install.html, You can guide them to apply for it on the official website iluvatar.com).


## Get ONNX model

The official documentation describes how to export a pytorch model as an onnx model. You can also download the onnx model from the official [repository](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime) or obtain from the data folder of the ixrt installation package.

## Model Conversion

Append implemented decoders and nms plugins.

```bash
cd samples/sampleYoloX
DATADIR=$(realpath ../../data/yolox_m)
# float16
python3 create_onnx.py --src ${DATADIR}/yolox_m.onnx --dest ${DATADIR}/yolox_m_with_decoder_nms.onnx
# int8 with qdq nodes
python3 create_onnx.py --src ${DATADIR}/yolox_m_qdq_quant.onnx --dest ${DATADIR}/yolox_m_qdq_with_decoder_nms.onnx --with_qdq
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
# float16
./build/bin/sampleYoloX --onnx ${DATADIR}/yolox_m_with_decoder_nms.onnx --engine ${DATADIR}/yolox_m_with_decoder_nms.engine --input ${DATADIR}/dog_640.jpg --plugin ./build/lib/liboss_ixrt_plugin.so
# int8 with qdq nodes
./build/bin/sampleYoloX --onnx ${DATADIR}/yolox_m_qdq_with_decoder_nms.onnx --engine ${DATADIR}/yolox_m_qdq_with_decoder_nms.engine --input ${DATADIR}/dog_640.jpg --type with_qdq --plugin ./build/lib/liboss_ixrt_plugin.so
```
