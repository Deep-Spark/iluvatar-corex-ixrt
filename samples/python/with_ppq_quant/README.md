# Use third party quantization tool with IxRT

In this sample, we demonstrate how to use IxRT API with third party quantization tool ([PPQ](https://github.com/openppl-public/ppq)). You can refer to [this](https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_OnnxParser.md) tutorial for more information.

## Prequisities
Here we demonstrate how to quantize a model with PPQ

1. Clone PPQ official repo and install PPQ

We tested on PPQ 0.6.6.
```bash
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
```

2. Prepare model and calibration dataset
```bash
mkdir -p working/data
```
Model
```bash
cp /path/to/your/model.onnx working/model.onnx
```

Put calibration images to `working/data/`, each image can be a binary buffer with suffix `.bin`

3. Quantization

Before running the command below, make sure in your script `ProgramEntrance_1.py`:
- `TARGET_PLATFORM   = TargetPlatform.TRT_INT8`
- `CALIBRATION_BATCHSIZE ` is less than then real number of calibration files you have, or scale could be wrong

```bash
python3 ProgramEntrance_1.py
```
After the script is executed, you will get 3 files in your working directory, `quantized.onnx`, `quant_cfg.json`, `quantized.wts`.
- `quantized.onnx` is is better for quantization that is used to deploy.
- `quant_cfg.json` contains quantization parameters.
- `quantized.wts` contains quantized weight parameters, if you want to deploy with trt.OnnxParser, please ignore it. But if you want to deploy with the api that comes with ixrt, please refer to [Define the model directly using the IxRT API](https://github.com/openppl-public/ppq/tree/master/md_doc/deploy_trt_by_api.md).

While in this sample, we are only interested in the first two files, copy them to this sample directory:
```
cp /path/to/ppq/working/{quantized.onnx,quant_cfg.onnx} /path/to/samples/python/with_ppq_quant
```

## Running the sample
Build engine
```bash
python write_qparams_onnx2ixrt.py \
    --onnx=quantized.onnx \
    --qparam_json=quant_cfg.json \
    --engine=int8-trt.engine
```
Here, the int8 engine has been built and ready for use.
