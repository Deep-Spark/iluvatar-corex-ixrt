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

from typing import List

from tabulate import tabulate as _tabulate

try:
    from tabulate import SEPARATING_LINE
except:
    SEPARATING_LINE = "\001"


def tabulate(values, headers=None, *args, **kwargs):
    if headers is None:
        headers = ["Name", "Value"]
        values = values.items()

    return _tabulate(values, headers=headers, *args, **kwargs)


def add_row_seperat_line(table: List):
    new_table = []
    for row in table:
        new_table.append(row)
        new_table.append(SEPARATING_LINE)
    return new_table
