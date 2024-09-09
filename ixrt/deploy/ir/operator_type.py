# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

from ..core.dynamic_enum import DynamicEnum


class OperatorType(DynamicEnum):
    ABS = "Abs"
    ACOS = "Acos"
    ACOSH = "Acosh"
    ADAPTIVE_AVG_POOL = "AdaptiveAvgPool"
    ADAPTIVE_MAX_POOL = "AdaptiveMaxPool"
    ADD = "Add"
    AND = "And"
    ARGMAX = "ArgMax"
    ARGMIN = "ArgMin"
    ASIN = "Asin"
    ASINH = "Asinh"
    ATAN = "Atan"
    ATANH = "Atanh"
    AVG_POOL = "AveragePool"
    BATCH_NORM = "BatchNormalization"
    BILINEAR = "Bilinear"
    CAST = "Cast"
    CAST_LIKE = "CastLike"
    CEIL = "Ceil"
    CELU = "Celu"
    CLIP = "Clip"
    CONCAT = "Concat"
    CONCAT_FROM_SEQUENCE = "ConcatFromSequence"
    CONV = "Conv"  # M
    CONV_TRANSPOSE = "ConvTranspose"
    COS = "Cos"
    COSH = "Cosh"
    CONSTANT = "Constant"
    CONSTANT_OF_SHAPE = "ConstantOfShape"
    DIV = "Div"
    DEQUANTIZELINEAR = "DequantizeLinear"
    QUANTIZELINEAR = "QuantizeLinear"
    DROPOUT = "Dropout"
    ELU = "Elu"
    EQUAL = "Equal"
    ERF = "Erf"
    EXP = "Exp"
    EXPAND = "Expand"
    EYE = "Eye"
    EYELIKE = "EyeLike"
    FLATTEN = "Flatten"
    FLOOR = "Floor"
    GATHER = "Gather"
    GATHER_ELES = "GatherElements"
    GATHER_ND = "GatherND"
    GELU = "Gelu"
    GEMM = "Gemm"
    GLOBAL_AVG_POOL = "GlobalAveragePool"
    GLOBAL_MAX_POOL = "GlobalMaxPool"
    GREATER = "Greater"
    GREATER_EQUAL = "GreaterOrEqual"
    GROUP_NORM = "GroupNormalization"
    GRU = "GRU"
    HARDSIGMOID = "HardSigmoid"
    HARDSHRINK = "HardShrink"
    HARDSWISH = "HardSwish"
    HARDMAX = "HardMax"
    HARDTANH = "HardTanh"
    IDENTITY = "Identity"
    INSTANCE_NORM = "InstanceNormalization"
    ISINF = "IsInf"
    ISNAN = "IsNaN"
    LAYER_NORM = "LayerNormalization"
    LEAKY_RELU = "LeakyRelu"
    LESS = "Less"
    LESS_EQUAL = "LessOrEqual"
    LRN = "LRN"
    LSTM = "LSTM"
    LOG = "Log"
    LOG_SIGMOID = "LogSigmoid"
    LOG_SOFTMAX = "LogSoftmax"
    MATMUL = "MatMul"
    MAX = "Max"
    MAX_POOL = "MaxPool"
    MAX_ROI_POOL = "MaxRoiPool"
    MAX_UNPOOL = "MaxUnpool"
    MEAN = "Mean"
    MIN = "Min"
    MISH = "Mish"
    MOD = "Mod"
    MUL = "Mul"
    NEG = "Neg"
    NONZERO = "NonZero"
    NOT = "Not"
    OR = "Or"
    PAD = "Pad"
    PERMUTE = "Permute"
    POW = "Pow"
    PRELU = "PRelu"
    RANDOM_UNIFORM_LIKE = "RandomUniformLike"
    RANGE = "Range"
    REDUCE_L1 = "ReduceL1"
    REDUCE_L2 = "ReduceL2"
    REDUCE_MAX = "ReduceMax"
    REDUCE_MEAN = "ReduceMean"
    REDUCE_MIN = "ReduceMin"
    REDUCE_PROD = "ReduceProd"
    REDUCE_SUM = "ReduceSum"
    RELU = "Relu"
    RRELU = "RRelu"
    RESHAPE = "Reshape"
    RESIZE = "Resize"
    RNN = "RNN"
    ROIALIGN = "RoiAlign"
    ROUND = "Round"
    SCATTER = "Scatter"
    SCATTER_ELES = "ScatterElements"
    SCATTER_ND = "ScatterND"
    GATHER_ND = "GatherND"
    SELU = "Selu"
    SEQUENCE_AT = "SequenceAt"
    SHAPE = "Shape"
    SHRINK = "Shrink"
    SLICE = "Slice"
    SIGMOID = "Sigmoid"
    SIGN = "Sign"
    SIN = "Sin"
    SINH = "Sinh"
    SILU = "Silu"
    SIZE = "Size"
    SOFTMAX = "Softmax"
    SOFTMIN = "Softmin"
    SOFTPLUS = "Softplus"
    SOFTSIGN = "Softsign"
    SPLIT = "Split"
    SPLIT_TO_SEQUENCE = "SplitToSequence"
    SQRT = "Sqrt"
    SQUEEZE = "Squeeze"
    SUB = "Sub"
    SUM = "Sum"
    TAN = "Tan"
    TANH = "Tanh"
    TILE = "Tile"
    TOPK = "TopK"
    TRANSPOSE = "Transpose"
    UPSAMPLE = "Upsample"
    UNSQUEEZE = "Unsqueeze"
    WHERE = "Where"
    YOLOV5_DECODER = "YoloV5Decoder"
    YOLOV7_DECODER = "YoloV7Decoder"

    SPEECH_ECAPA_POOL = "EcapaPool"
    SPEECH_ECAPA_ASP_ATTN = "EcapaAspAttn"
    SPEECH_ECAPA_SCORE_POOL = "EcapaScorePool"

    # 使用 FX 构建图时的类型
    GETATTR = "getattr"
    GETITEM = "getitem"


SKIP_SIMULATIVE_QUANT_OPERATORS = [
    OperatorType.ARGMAX,
    OperatorType.ARGMIN,
    OperatorType.CONSTANT_OF_SHAPE,
    OperatorType.DROPOUT,
    OperatorType.EXPAND,
    OperatorType.EYE,
    OperatorType.EYELIKE,
    OperatorType.IDENTITY,
    OperatorType.ISINF,
    OperatorType.ISNAN,
    OperatorType.PAD,
    OperatorType.RANGE,
    OperatorType.RESIZE,
    OperatorType.SHAPE,
    OperatorType.SIZE,
    OperatorType.TILE,
    OperatorType.TOPK,
]


def get_support_ops():
    for name in OperatorType.__dict__:
        name: str
        if name.isupper():
            print(f'    ("{getattr(OperatorType, name)}", ),')
