# 分类模型量化

## 描述

本目录下提供了两个对分类模型进行量化的脚本，这两个脚本分别采用了PTQ(ptq_classification.py)和QAT(qat_classification.py)的方式。

## 支持的模型

|  | ResNet18 | ResNet50 | MobileNet v2 | ResNext50 32x4d | VGG16 BN | SqueezeNet 1.0x | MnasNet 1.0x | ShuffleNet v2 1.0x | RegNet Y 400MF |
| :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  | :----:  | :----:  | :----:  | :----:  |
| PTQ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| QAT | √ |   |   |   |   |   |   |   |   |

Note: QAT目前只测试了ResNet18

## 参数及作用

| 参数 | 作用 |
| :----:  |  :----:  |
| --batch_size | 指定batchsize，default:64 |
| --img_size | 指定图片大小，default:224 |
| --workers | 加载数据集，default:4 |
| --model | 指定模型（名称或者ONNX路径） |
| --num_samples | 指定校准数据集的大小 |
| --data_path | 指定数据集路径 |
| --analyze | 误差分析 |
| --observer | 指定observer类型,default:"hist_percentile",目前支持的observer类型：minmax, hist_percentile, percentile, entropy, ema |
| --fp32_acc | 计算fp32精度 |
| --use_ixrt | 使用ixrt进行推理 |
| --quant_params | 指定量化参数文件的路径，若指定量化参数，则只验证，不进行量化 |
| --disable_bias_correction | 禁用误差校正 |

## 运行命令

```bash
# 指定模型名称

python3 ptq_classification.py --model resnet18 --data_path /path/to/imagenet/val

# 指定ONNX模型路径

python3 ptq_classification.py --model /path/to/resnet18.onnx --data_path /path/to/imagenet/val

```
