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

import inspect
from collections import OrderedDict


class Registry(object):
    def __init__(self, name=None, instantiate_cls=False):
        self.name = name
        self.handlers = OrderedDict()
        self.instantiate_cls = instantiate_cls

    def add_handler(self, key, handler):
        self.handlers[key] = handler

    def get_handlers(self):
        return self.handlers.values()

    def get(self, name, **kwargs):
        if "default" in kwargs:
            return self.handlers.get(name, kwargs["default"])
        return self.handlers[name]

    def containe(self, name):
        return name in self.handlers

    def registe(self, name=None, alias=None):
        def wrap(handler):
            handler_name = name
            if handler_name is None:
                if hasattr(handler, "__name__"):
                    handler_name = handler.__name__
                else:
                    handler_name = handler.__class__.__name__

            if self.instantiate_cls and inspect.isclass(handler):
                handler = handler()

            self.add_handler(handler_name, handler)
            if alias is not None:
                self.add_handler(alias, handler)
            return handler

        return wrap


def check_build_config(cfg, registry, default_args):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args must be a dict or None, " f"but got {type(default_args)}"
        )


def get_type_from_cfg(cfg):
    return cfg["type"]


def build_from_cfg(cfg, registry, default_args=None, check_cfg=True):
    """Build a module from config dict.

    Parameters
    ----------
    cfg : dict
        Config dict. It should at least contain the key "type".
    registry : :obj:`Registry`
        The registry to search the type from.
    default_args : dict, optional
        Default initialization arguments.

    Returns
    -------
    module_obj : object
        The constructed object.
    """
    if check_cfg:
        check_build_config(cfg, registry, default_args)

    args = cfg.copy()
    module_type = args.pop("type")
    if isinstance(module_type, str):
        module_obj = registry.get(module_type)
        if module_obj is None:
            raise KeyError(f"{module_type} is not in the {registry.name} registry")
    elif inspect.isclass(module_type):
        module_obj = module_type
    else:
        raise TypeError(
            f"type must be a str or valid type, but got {type(module_type)}"
        )

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return module_obj(**args)
