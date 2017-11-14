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
import numpy as np
from ngraph.util.names import NameableValue


class Dtype(NameableValue):
    families = ["bool", "int", "float"]
    dtypes = {}

    def __init__(self,
                 name,
                 numpy_dtype,
                 family=None,
                 **kwargs):
        super(Dtype, self).__init__(name=name, **kwargs)
        self.__numpy_dtype = numpy_dtype
        if family is None:
            family = self.deduce_family()
        self.__family = family

        if family not in Dtype.families:
            raise ValueError("Unrecognized family %s" % family)

        Dtype.dtypes[self.name] = self

    def deduce_family(self):
        for family in Dtype.families:
            if family in self.name:
                return family
        raise ValueError("Could not deduce family from name: %s" % self.name)

    @property
    def numpy_dtype(self):
        return self.__numpy_dtype

    @property
    def itemsize(self):
        return self.__numpy_dtype.itemsize

    @property
    def family(self):
        return self.__family

    def lub(self, other):
        my_family_idx = Dtype.families.index(self.family)
        other_family_idx = Dtype.families.index(other.family)

        my_key = (my_family_idx, self.itemsize, self.name)
        other_key = (other_family_idx, other.itemsize, other.name)

        if my_key < other_key:
            return other
        elif my_key > other_key:
            return self
        else:
            assert self is other
            return self

    def __str__(self):
        return "Dtype(" + self.name + ")"

    def __repr__(self):
        return str(self)


Float32 = Dtype("float32", np.dtype("float32"))
Float64 = Dtype("float64", np.dtype("float64"))


def to_dtype(x):
    """
    Interprets an object as an ngraph dtype.
    """
    if isinstance(x, str):
        return Dtype.dtypes[x]
    elif isinstance(x, np.dtype):
        return Dtype.dtypes[x.name]
    elif isinstance(x, Dtype):
        return x
    else:
        raise ValueError("Unrecognized type")
