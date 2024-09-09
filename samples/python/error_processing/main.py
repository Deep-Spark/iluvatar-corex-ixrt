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

import ixrt


class MyErrorRecorder(ixrt.IErrorRecorder):
    def __init__(self):
        self.errors = []
        super().__init__()

    def report_error(self, val, desc):
        self.errors.append((val, desc))

    def num_errors(self):
        return len(self.errors)

    def get_error_code(self, idx):
        if idx < self.num_errors():
            return self.errors[idx][0]
        else:
            raise IndexError(
                f"Wrong index {idx}, which is out of [0, {self.num_errors()}]"
            )

    def get_error_desc(self, idx):
        if idx < self.num_errors():
            return self.errors[idx][1]
        else:
            raise IndexError(
                f"Wrong index {idx}, which is out of [0, {self.num_errors()}]"
            )

    def has_overflowed(self):
        return False

    def clear(self):
        self.errors.clear()


error_recorder1 = MyErrorRecorder()
logger = ixrt.Logger(ixrt.Logger.VERBOSE)


def sample_dealwith_error(engine_path):
    with open(engine_path, "rb") as f, ixrt.Runtime(logger) as runtime:
        runtime.error_recorder = error_recorder1
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        for i in range(error_recorder1.num_errors()):
            print(f"#{i} error code:", error_recorder1.get_error_code(i))
            print(f"#{i} error desc:", error_recorder1.get_error_desc(i))
        assert not engine


sample_dealwith_error("/bin/ls")
