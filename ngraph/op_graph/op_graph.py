# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import numpy as np
from ngraph.util.names import ScopedNameableValue
from ngraph.op_graph.axes import Axes, OrderedAxes
from ngraph.op_graph.dtype import to_dtype


class Op(ScopedNameableValue):
    def __init__(self,
                 args=(),
                 device="",
                 **kwargs):
        self.__args = args
        self.__device = device
        self.__control_deps = set()

    @property
    def args(self):
        return self.__args

    @property
    def device(self):
        return self.__device

    @property
    def control_deps(self):
        return self.__control_deps

    def __repr__(self):
        return "<{cl}:{id}>".format(
            cl=self.__class__.__name__,
            id=id(self)
        )


class TensorOp(Op):
    def __init__(self,
                 axes=None,
                 ordered_axes=None,
                 is_persistent=False,
                 is_constant=False,
                 dtype=None,
                 **kwargs):
        super(TensorOp, self).__init__(**kwargs)
        self.__axes = Axes(axes)
        if ordered_axes is None:
            ordered_axes = collections.OrderedDict(axes)
        self.__ordered_axes = OrderedAxes(ordered_axes)
        self.__is_persistent = is_persistent
        self.__is_constant = is_constant
        self.__dtype = to_dtype(dtype)

    @property
    def axes(self):
        return self.__axes

    @property
    def ordered_axes(self):
        return self.__ordered_axes

    @property
    def is_persistent(self):
        return self.__is_persistent

    @property
    def is_constant(self):
        return self.__is_constant

    @property
    def dtype(self):
        return self.__dtype


class ConstOp(TensorOp):
    def __init__(self,
                 const,
                 **kwargs):
        super(ConstOp, self).__init__(**kwargs)
        self.__const = const

    @property
    def const(self):
        return self.__const


def constant(const, ordered_axes=None, dtype=None, **kwargs):
    """
    Makes a constant scalar/tensor.  For a tensor, constant provides the opportunity
        to supply axes.  Scalar/NumPytensor arguments are usually automatically converted to
        tensors, but constant may be used to supply axes or in the rare cases where constant
        is not automatically provided.

    Args:
        const: The constant, a scalar or a NumPy array.
        axes: The axes for the constant.
        dtype (optional): The dtype to use.
    Returns:
        A ConstOp for the constant.
    """

    nptensor = np.asarray(const)
    if not ordered_axes:
        ordered_axes = {"ax{}".format(l): l for l in nptensor.shape}
    ordered_axes = OrderedAxes(ordered_axes)
    axes = Axes(ordered_axes)

    if len(axes) != len(nptensor.shape):
        raise ValueError("The axes do not match the shape of the constant.")

    if dtype is None:
        dtype = nptensor.dtype
    dtype = to_dtype(dtype)
    nptensor = nptensor.astype(dtype.numpy_dtype)

    val = ConstOp(nptensor,
                  axes=axes,
                  ordered_axes=ordered_axes,
                  dtype=dtype,
                  **kwargs)
    return val


class ElementWiseOp(TensorOp):
    pass


class BinaryElementWiseOp(ElementWiseOp):
    def __init__(self, x, y, **kwargs):
        if x.axes != y.axes:
            raise ValueError("The axes of the arguments do not match.")
        super(BinaryElementWiseOp, self).__init__(args=(x, y),
                                                  axes=x.axes,
                                                  dtype=x.dtype.lub(y.dtype),
                                                  **kwargs)


class Add(BinaryElementWiseOp):
    pass


def add(*args):
    return Add(*args)


class Multiply(BinaryElementWiseOp):
    pass


def multiply(*args):
    return Multiply(*args)
