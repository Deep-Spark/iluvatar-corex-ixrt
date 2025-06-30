## 0. Introduction
This is a quantization sample on Iluvatar GPU, using pytorch-quantization tool.
You can export any model registered in `torchvision` with this script.

## 1. Dependencies
### Dependent software libraries
```bash
# 官方库依赖 libcudart.so.12，天数目前只有 libcudart.so.10.2，无法直接使用，需要编译安装源码
# pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com pytorch-quantization==2.2.1

git clone ssh://git@bitbucket.iluvatar.ai:7999/swapp/pytorch-quantization.git
cd pytorch-quantization
python3 setup.py install
pip3 install tqdm torchvision
```
### Dependent data
ImageNet val is used in this sample, but you can also use your own validate data


## 2. Quantization and quick inference test with IxRT
```bash
python3 quant.py --batch_size 512 --data_path /path/to/image/val --model vgg16
```
If no error happens, a QDQ format onnx file `quant_<model>.onnx` is on the current directory.

Then you can have a quick build engine test and inference with `ixrtexec`

```bash
ixrtexec --onnx quant_vgg16.onnx --precision int8 --log_level verbose
```
