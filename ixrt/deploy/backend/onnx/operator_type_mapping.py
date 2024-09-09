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

ONNX_OPS = [
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AttributeHasValue",
    "AveragePool",
    "BatchNormalization",
    "Bernoulli",
    "BitShift",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "BlackmanWindow",
    "Cast",
    "CastLike",
    "Ceil",
    "Celu",
    "CenterCropPad",
    "Clip",
    "Col2Im",
    "Compress",
    "Concat",
    "ConcatFromSequence",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "Cos",
    "Cosh",
    "CumSum",
    "DFT",
    "DepthToSpace",
    "DequantizeLinear",
    "Det",
    "Div",
    "Dropout",
    "DynamicQuantizeLinear",
    "Einsum",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "EyeLike",
    "Flatten",
    "Floor",
    "GRU",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gemm",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "GridSample",
    "HammingWindow",
    "HannWindow",
    "HardSigmoid",
    "HardSwish",
    "Hardmax",
    "Identity",
    "If",
    "InstanceNormalization",
    "IsInf",
    "IsNaN",
    "LRN",
    "LSTM",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "Loop",
    "LpNormalization",
    "LpPool",
    "MatMul",
    "MatMulInteger",
    "Max",
    "MaxPool",
    "MaxRoiPool",
    "MaxUnpool",
    "Mean",
    "MeanVarianceNormalizaton",
    "MelWeightMatrix",
    "Min",
    "Mish",
    "Mod",
    "Mul",
    "Multinomial",
    "Neg",
    "NegativeLogLikelihoodLss",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "RNN",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Range",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "RoiAlign",
    "Round",
    "STFT",
    "Scan",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Selu",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SequenceMap",
    "Shape",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Size",
    "Slice",
    "Softmax",
    "SoftmaxCrossEntropyLos",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Split",
    "SplitToSequence",
    "Sqrt",
    "Squeeze",
    "StringNormalizer",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "TfIdfVectorizer",
    "ThresholdedRelu",
    "Tile",
    "TopK",
    "Transpose",
    "Trilu",
    "Unique",
    "Unsqueeze",
    "Upsample",
    "Where",
    "Xor",
]


ONNX2IR = dict()


UNSUPPORTED_OPS = [
    "AttributeHasValue",
    "Bernoulli",
    "BitShift",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "BlackmanWindow",
    "CenterCropPad",
    "Col2Im",
    "Compress",
    "ConvInteger",
    "CumSum",
    "DFT",
    "DepthToSpace",
    "DequantizeLinear",
    "Det",
    "DynamicQuantizeLinear",
    "Einsum",
    "GlobalLpPool",
    "GridSample",
    "HammingWindow",
    "HannWindow",
    "If",
    "Loop",
    "LpNormalization",
    "LpPool",
    "MatMulInteger",
    "MeanVarianceNormalizaton",
    "MelWeightMatrix",
    "Multinomial",
    "NegativeLogLikelihoodLss",
    "NonMaxSuppression",
    "OneHot",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "Reciprocal",
    "ReverseSequence",
    "STFT",
    "Scan",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SequenceMap",
    "SoftmaxCrossEntropyLoss",
    "SpaceToDepth",
    "StringNormalizer",
    "TfIdfVectorizer",
    "ThresholdedRelu",
    "Trilu",
    "Unique",
    "ReduceSumSquare",
    "Xor",
]


def onnx2ir_op(op_name: str):
    if op_name in ONNX2IR:
        return ONNX2IR[op_name]
    return op_name


def ir2onnx_op(op_name: str):
    for onnx_op, ir_op in ONNX2IR.items():
        if ir_op == op_name:
            return onnx_op

    return op_name
