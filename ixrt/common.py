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

import numpy as np
import ixrt._C as dtype

from ._C import PipelineType
from ._C import RuntimeConfig as RuntimeConfig_
from ._C import RuntimeContext as _RuntimeContext
from ._C import TensorShape, TensorShapeMap, device
from ._C import format as dformat

allowed_dtypes = ["int8", "int32", "int64", "float16", "float32", "float64", "bool"]
allowed_data_format = [
    "linear",
    "nhwc",
]


class RuntimeConfig(RuntimeConfig_):
    def __repr__(self):
        s = f"""RuntimeConfig(
    input_shapes = {self.input_shapes},
    device_idx = {self.device_idx},
    graph_file = {self.graph_file},
    weights_file = {self.weights_file},
    quant_file = {self.quant_file},
    runtime_context = {RuntimeContext.__repr__(self.runtime_context)}
)"""
        return s

    def __str__(self):
        return self.__repr__()


class RuntimeContext(_RuntimeContext):
    def __init__(
        self,
        data_type: str or np.dtype = "int8",
        data_format: str = "linear",
        use_gpu: bool = True,
        pipeline_sync: bool = True,
        input_types: dict = dict(),
        output_types: dict = dict(),
        input_device: str = "cpu",
        output_device: str = "cpu",
    ):
        """
        Args:
            data_type: General data type of inference, str or np.dtype
            data_format: Format of input data, str
            use_gpu: Use gpu to infer or not, bool
            pipeline_sync: Inference in synchronous or asynchronous way, bool
            input_types: Types of input data, dict[string, dtype]
            output_types: Types of output data, dict[string, dtype]
        """
        # parameters check
        data_type = str(np.dtype(data_type))
        if type(data_type) not in [str, np.dtype]:
            raise ValueError(
                f"data_type should be either str or np.dtype, but got {type(data_type)}"
            )
        if data_type not in allowed_dtypes:
            raise ValueError(f"data_type should be one of {allowed_dtypes}")
        if data_format not in allowed_data_format:
            raise ValueError(
                f"data_format should be one of {allowed_data_format}, not {data_format}"
            )
        if not isinstance(use_gpu, bool):
            raise ValueError(f"target_device should be bool, not {type(use_gpu)}")
        if not isinstance(pipeline_sync, bool):
            raise ValueError(
                f"pipeline_sync should be bool value, not {type(pipeline_sync)}"
            )
        if not pipeline_sync:
            raise ValueError(
                f"pipeline support PIPELINE_TYPE_SYNCHRONOUS mode only, PIPELINE_TYPE_ASYNCHRONOUS mode has been deprecated."
            )

        def to_dict_string(dict_ob):
            result = [
                "'{}':dtype.bool".format(k)
                if v == "bool"
                else "'{}':dtype.{}".format(k, v)
                for k, v in dict_ob.items()
            ]
            result = ",".join(result)
            return "{" + result + "}"

        _args = [
            "dtype.bool" if data_type == "bool" else "dtype.{}".format(data_type),
            "dformat.{}".format(data_format),
            "device.{}".format("gpu" if use_gpu else "cpu"),
            "PipelineType.PIPELINE_TYPE_{}".format(
                "SYNCHRONOUS" if pipeline_sync else "ASYNCHRONOUS"
            ),
            to_dict_string(input_types),
            to_dict_string(output_types),
            "device.{}".format(input_device.lower()),
            "device.{}".format(output_device.lower()),
        ]
        super().__init__(*map(eval, _args))

    def set_dtype(self, _dtype: str):
        if _dtype not in allowed_dtypes:
            raise ValueError(
                f"data_type should be either str or np.dtype, but got {type(dtype)}"
            )
        self.dtype = getattr(dtype, _dtype)

    def set_data_format(self, format: str):
        if format not in allowed_data_format:
            raise ValueError(
                f"data_format should be one of {allowed_data_format}, not {format}"
            )
        self.format = getattr(dformat, format)

    def set_input_types(self, new_dtypes: dict):
        _types = dict()
        for k, v in new_dtypes.items():
            _types[k] = dtype.__dict__[v]
        self.input_types = _types

    def set_output_types(self, new_dtypes: dict):
        _types = dict()
        for k, v in new_dtypes.items():
            _types[k] = dtype.__dict__[v]
        self.output_types = _types

    def set_pipeline_mode(self, sync):
        if sync:
            self.pipeline_type = PipelineType.PIPELINE_TYPE_SYNCHRONOUS
        else:
            raise ValueError(
                f"PIPELINE_TYPE_SYNCHRONOUS mode only, PIPELINE_TYPE_ASYNCHRONOUS mode has been deprecated."
            )

    def __repr__(self):
        return (
            f"RuntimeContext("
            f"dtype={self.dtype}, "
            f"format={self.format}, "
            f"device_type={self.device_type}, "
            f"pipeline_type={self.pipeline_type}, "
            f"input_types={self.input_types}, "
            f"output_types={self.output_types}, "
            f"input_device={self.input_device}, "
            f"output_device={self.output_device}"
            f")"
        )

    def __str__(self):
        return self.__repr__()
