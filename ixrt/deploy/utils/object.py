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
from typing import Any, Callable, Dict, Mapping, Union

try:
    from .fundoc import FunctionDoc
except:
    FunctionDoc = None


__all__ = [
    "isfunction",
    "iscallable",
    "get_obj_name",
    "isimmutable_var",
    "get_self_from",
    "get_obj_funcs",
    "recurse_getattr",
    "recurse_find_by_key",
    "set_value_by_cascasde_key",
    "flatten_container",
    "flatten_dict",
    "get_func_argspec",
    "get_obj_attr",
    "get_function_param_names",
    "get_namedtuple_fields",
    "get_namedtuple_defaults",
    "isnamedtuple",
    "namedtype_to_dict",
]


def isfunction(f):
    return inspect.isfunction(f) or inspect.ismethod(f) or inspect.isbuiltin(f)


def iscallable(fn) -> bool:
    return any(
        [
            callable(fn),
            inspect.isfunction(fn),
            inspect.ismethod(fn),
            inspect.isbuiltin(fn),
        ]
    )


def get_obj_name(obj):
    if hasattr(obj, "__name__"):
        return obj.__name__

    if inspect.isclass(obj):
        return obj.__class__.__name__

    return str(obj)


def isimmutable_var(var):
    if var is None:
        return True

    if inspect.isclass(var):
        var_cls = var
    else:
        var_cls = type(var)

    return var_cls in [int, float, tuple, str, None]


def get_self_from(obj):
    if hasattr(obj, "__self__"):
        return obj.__self__
    raise AttributeError(f"Not found attribute `self` in {obj}.")


def get_obj_funcs(obj) -> Dict[str, Callable]:
    attrs = dir(obj)
    funcs = dict()
    for attr in attrs:
        fn = getattr(obj, attr)
        if iscallable(fn):
            funcs[attr] = fn

    return funcs


def recurse_find_by_key(container: dict, key: Union[str, list], default=None):
    if isinstance(key, str):
        key = key.split(".")

    if not isinstance(key, (tuple, list)):
        raise RuntimeError(f"Please give the type str or list, but get ({type(key)}).")

    value = default
    _cnt = container
    for k in key:
        if k not in _cnt:
            return default
        value = _cnt[k]
        _cnt = value
        if _cnt is None:
            return default

    return value


def set_value_by_cascasde_key(container: dict, key: str, value: Any):
    if isinstance(key, str):
        key = key.split(".")

    if not isinstance(key, (tuple, list)):
        raise RuntimeError(f"Please give the type str or list, but get ({type(key)}).")

    _cnt = container
    for k in key[:-1]:
        if k not in _cnt:
            _cnt[k] = dict()
        _cnt = _cnt[k]
    _cnt[key[-1]] = value
    return container


def flatten_dict(d: dict, preffix="", out=None):
    if out is None:
        out = dict()
    for k, v in d.items():
        if isinstance(v, Mapping):
            flatten_dict(v, f"{preffix}{k}.", out)
        else:
            out[preffix + k] = v

    return out


def flatten_container(container: Union[list, dict]):
    outs = []

    def _flatten_list(cnt: list):
        for item in cnt:
            if isinstance(item, (tuple, list)):
                _flatten_list(item)
            elif isinstance(item, Mapping):
                _flatten_dict(item)
            else:
                outs.append(item)

    def _flatten_dict(cnt: Dict):
        for key, item in cnt.items():
            if isinstance(item, (tuple, list)):
                _flatten_list(item)
            elif isinstance(item, Mapping):
                _flatten_dict(item)
            else:
                outs.append(item)

    if isinstance(container, (tuple, list)):
        _flatten_list(container)
    elif isinstance(container, dict):
        _flatten_dict(container)
    else:
        outs.append(container)

    return outs


def get_func_argspec(func) -> inspect.FullArgSpec:
    return inspect.getfullargspec(func)


def get_obj_attr(obj, attr, default=None):
    if isinstance(obj, Mapping):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def get_function_param_names(function):
    try:
        return inspect.getfullargspec(function).args
    except:
        pass
    try:
        summary = FunctionDoc(function)["Summary"][0]
        start_idx = summary.index("(") + 1
        end_idx = summary.index(")")
        params = summary[start_idx:end_idx]
        params = params.split(",")
    except:
        raise RuntimeError(f"Not fetch the parameters of {function}.")

    ret = []
    for param in params:
        param = param.split("=")[0].lstrip().rstrip()
        ret.append(param)
    return ret


def isnamedtuple(obj):
    if not inspect.isclass(obj) or not issubclass(obj, tuple):
        return False

    if hasattr(obj, "_fields") and hasattr(obj, "_replace"):
        if (
            hasattr(obj._replace, "__module__")
            and obj._replace.__module__ == "collections"
        ):
            return True

    return False


def get_namedtuple_fields(t):
    if not inspect.isclass(t):
        t = type(t)

    if not isnamedtuple(t):
        raise RuntimeError(f"{t} is not a namedtuple object")

    return t._fields


def get_namedtuple_defaults(t) -> dict:
    return t._field_defaults


def namedtype_to_dict(t):
    return t._asdict()


def recurse_getattr(obj, attr: str, sep="."):
    attrs = attr.split(sep)
    idx = 0
    cur_obj = obj
    while idx < len(attrs):
        cur_obj = getattr(cur_obj, attrs[idx])
        idx += 1

    if cur_obj == obj:
        return None
    return cur_obj
