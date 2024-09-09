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

def green(s):
    return "\033[32m" + s + "\033[0m"


def red(s):
    return "\033[31m" + s + "\033[0m"


def colorize(s, condition):
    if condition:
        return green(str(s))
    else:
        return red(str(s))


def num2str(n, _range: tuple):
    assert len(_range) == 2
    lhs, rhs = _range[0], _range[1]
    if rhs < lhs:
        condition = True
    else:
        condition = lhs <= n <= rhs
    return colorize(str(n), condition)
