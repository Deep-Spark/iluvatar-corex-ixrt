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

import os

_TRUE_LIST = ["t", "true", "1", "on"]


def _get_bool_from_env(name):
    return os.environ.get(name, "F").lower() in _TRUE_LIST


def is_static_convert():
    return _get_bool_from_env("STATIC_CONVERT")


def is_force_export_opeator_onnx():
    return _get_bool_from_env("FORCE_EXPORT_OPERATOR_ONNX")
