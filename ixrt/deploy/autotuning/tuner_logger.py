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

import logging
from functools import wraps

tuner_logger: logging.Logger = None


def get_tuner_logger():
    global tuner_logger
    if tuner_logger is None:
        tuner_logger = logging.getLogger("TunerLogger")
        logging.basicConfig(
            level=logging.INFO,
            format="[AutoTuning] %(levelname)s %(message)s",
        )

    return tuner_logger
