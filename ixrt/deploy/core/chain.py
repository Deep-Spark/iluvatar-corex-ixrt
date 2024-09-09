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

from abc import abstractmethod
from typing import Any, Callable, List, NamedTuple, Union


class Result(NamedTuple):
    status: bool
    value: Any = None


class ChainHandler(object):
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> Result:
        pass


class Chain(ChainHandler):
    def __init__(
        self, handlers: List[Union[Callable, ChainHandler]] = None, default: Any = None
    ):
        self.handlers = handlers or []
        self.default = default

    def add_handler(self, handler, index=None):
        index = len(self.handlers) if index is None else index
        self.handlers.insert(index, handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def run(self, *args, **kwargs) -> Result:
        for handler in self.handlers:
            response = handler(*args, **kwargs)
            if response is not None:
                if isinstance(response, Result):
                    if response.status:
                        return response
                else:
                    return response

        # No handlers to handle this event
        return Result(False, self.default)

    def handler(self, index=None):
        def _add_handler(handler):
            self.add_handler(handler, index=index)
            return handler

        return _add_handler
