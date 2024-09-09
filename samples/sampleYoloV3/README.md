# YoloV3 INT8 Inference Sample
## Prequisities

### Test data
Download the sample data from the IxRT release tarball.

### Install IxRT library
Follow steps from IxRT's official document.

## Running the sample
0. Quantization
With the help of IxRT deploy tool, quantize it to INT8 QDQ model
```bash
DATADIR=$(realpath data/yolov3)
python3 samples/sampleYoloV3/yolov3_qdq.py --model ${DATADIR}/yolov3_without_decoder.onnx --save_model_path ${DATADIR}/yolov3_without_decoder_qdq.onnx
```
1. Modify the YoloV3 onnx to specified needs

Assume you are in the root of IxRT oss, execute commands below:
```bash
# Use IxRT plugin: YoloV3DecoderPlugin_IxRT
python3 samples/sampleYoloV3/deploy.py --src ${DATADIR}/yolov3_without_decoder_qdq.onnx --dest ${DATADIR}/yolov3_qdq_plugin.onnx --custom
# Use dynamic input and plugin
python3 samples/sampleYoloV3/deploy.py --src ${DATADIR}/yolov3_without_decoder_qdq.onnx --dest ${DATADIR}/yolov3_qdq_plugin_dynamic.onnx --custom --dynamic
```
Arguments:
- `--src`, the input yolov3 onnx without decoders
- `--dest`, the output yolov3 onnx with decoders
- `--custom`, if specified, use IxRT plugin
- `--dynamic`, if specified, use dynamic shape for H/W of the YoloV3 input

2. Compile the samples in IxRT OSS directory
If you haven't compiled the IxRT samples, following command helps you compile all samples
```
cd path-to-ixrt-oss
cmake -B build
cmake --build build -j
```

3. Run
```bash
# Execute with IxRT new API, this example will call plugin YoloV3DecoderPlugin_IxRT
./build/bin/sampleYoloV3 --onnx ${DATADIR}/yolov3_qdq_plugin.onnx --engine ${DATADIR}/yolov3_qdq_plugin.engine --demo trt_exe
# Execute with IxRT new API, dynamic shape, this example will call plugin YoloV3DecoderPlugin_IxRT
./build/bin/sampleYoloV3 --onnx ${DATADIR}/yolov3_qdq_plugin_dynamic.onnx --engine ${DATADIR}/yolov3_qdq_plugin_dynamic.engine --demo trt_dyn
```
