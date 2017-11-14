# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# ----------------------------------------------------------------------------
import collections


def validate_and_convert(x):
    if not isinstance(x, collections.Mapping):
        raise ValueError("The input must be a mapping.")

    if not isinstance(x, collections.Iterable):
        x = collections.OrderedDict(x)

    for key, value in x.items():
        if not isinstance(key, str):
            raise ValueError("The key {} is not a string.".format(key))
        if value is not None and not isinstance(value, int):
            raise ValueError("The value {} is not an integer or None.".format(value))
    return x


class Axes(dict):
    def __init__(self, x=None, **kwargs):
        if x is None:
            super(Axes, self).__init__(**kwargs)
        else:
            super(Axes, self).__init__(validate_and_convert(x), **kwargs)

    def __setitem__(self, item):
        raise ValueError("Axes are immutable")


class OrderedAxes(collections.OrderedDict):
    def __init__(self, x=None, **kwargs):
        if x is None:
            super(OrderedAxes, self).__init__(**kwargs)
        else:
            super(OrderedAxes, self).__init__(validate_and_convert(x), **kwargs)

    def __setitem__(self, item):
        raise ValueError("Ordered axes are immutable")
