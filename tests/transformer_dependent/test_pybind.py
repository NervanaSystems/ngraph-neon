#!/usr/bin/env python
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

from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt
import numpy as np


def test_add_with_scalar():

    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=4, name='width')
    a = ng.placeholder(axes=[H, W])

    d = ng.add(a, 5)
    available_transformer = ngt.transformer_choices()

    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            # Define a computation
            _add = pybind_exec.computation(d, a)
            d_val = _add([10, 20, 30, 40])

            # compute reference through numpy
            d_val_ref = np.add(np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 4),
                               np.array([5], dtype=np.float32))
            assert np.allclose(d_val[0], d_val_ref)
    else:
        raise AssertionError("Unable to initialize pybind_translator transformer")


def test_add_with_mul():

    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=1, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])
    c = ng.placeholder(axes=[H, W])
    d = ng.multiply(ng.add(a, b), c)

    available_transformer = ngt.transformer_choices()

    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            # Define a computation
            _mul = pybind_exec.computation(d, a, b, c)
            d_val = _mul([10], [20], [10])

            # compute reference through numpy
            _add_ref = np.add(np.full([1, 1], 10, dtype=np.float32),
                              np.full([1, 1], 20, dtype=np.float32))
            d_val_ref = np.multiply(_add_ref, np.full([1, 1], 10, dtype=np.float32))
            assert np.allclose(d_val, d_val_ref)
    else:
        raise AssertionError("Unable to initialize pybind_translator transformer")


def test_multiple_computation():
    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=1, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])

    _mul = ng.multiply(a, b)
    _add = ng.add(a, b)

    available_transformer = ngt.transformer_choices()

    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            # Define a computation
            _mul_computation = pybind_exec.computation(_mul, a, b)
            _mul_val = _mul_computation([10], [20])
            _add_computation = pybind_exec.computation(_add, a, b)
            _add_val = _add_computation([10], [20])

            # compute reference vlue
            _mul_ref = np.multiply(np.full([1, 1], 10, dtype=np.float32),
                                   np.full([1, 1], 20, dtype=np.float32))
            _add_ref = np.add(np.full([1, 1], 10, dtype=np.float32),
                              np.full([1, 1], 20, dtype=np.float32))

            assert np.allclose(_add_val, _add_ref)
            assert np.allclose(_mul_val, _mul_ref)
    else:
        raise AssertionError("Unable to initialize pybind_translator transformer")


def test_Add_with_muliple_axis():
    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=4, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])

    d = ng.add(a, b)
    available_transformer = ngt.transformer_choices()

    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            # Define a computation
            _add = pybind_exec.computation(d, a, b)
            d_val = _add([10, 20, 30, 40], [11, 12, 13, 14])

            # compute reference through numpy
            d_val_ref = np.add(np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 4),
                               np.array([11, 12, 13, 14], dtype=np.float32).reshape(1, 4))

            assert np.allclose(d_val, d_val_ref)
    else:
        raise AssertionError("Unable to initialize pybind_translator transformer")


def test_broadcast():
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=4)

    np_a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    np_c = np.add(np_a, 2)

    a = ng.constant(np_a, [M, N])
    c = ng.add(a, 2)

    available_transformer = ngt.transformer_choices()

    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            # Define a computation
            _add = pybind_exec.computation(c)
            result = _add()

            assert np.allclose(result, np_c)
    else:
        raise AssertionError("Unable to initialize pybind_translator transformer")


def test_dot():
    H = ng.make_axis(length=1)
    W = ng.make_axis(length=4)
    np_a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    np_b = np.array(3, dtype=np.float32)

    a = ng.constant(np_a, [H, W])
    b = ng.constant(np_b, [])
    c = ng.dot(a, b)

    available_transformer = ngt.transformer_choices()
    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            _dot = pybind_exec.computation(c)
            _dot_val = _dot()

            # compute reference
            _dot_val_ref = np.dot(np_a, np_b)

            # this checks the dot product between scalar and vector, this is equivalent to
            # elementwise multiplication between scalar and vector
            assert np.allclose(_dot_val, _dot_val_ref)


def test_sum():

    H = ng.make_axis(length=2)
    W = ng.make_axis(length=2)
    H1 = ng.make_axis(length=1)
    W1 = ng.make_axis(length=4)

    input1 = ng.placeholder(axes=[H, W])
    input2 = ng.placeholder(axes=[H1, W1])

    # does reduction sum operation along axis[0]:H
    sum_op_1 = ng.sum(input1, reduction_axes=H)

    # sum elements across all the axis
    sum_op_2 = ng.sum(input2)
    available_transformer = ngt.transformer_choices()
    if "pybind_translator" in available_transformer:
        with closing(ngt.make_transformer_factory('pybind_translator',
                                                  backend="INTERPRETER")()) as pybind_exec:
            _sum = pybind_exec.computation(sum_op_1, input1)
            _sum_val = _sum([[1, 2], [3, 4]])
            assert np.array_equal(_sum_val, [4, 6])

            _sum = pybind_exec.computation(sum_op_2, input2)
            _sum_val = _sum([1, 2, 3, 4])
            assert np.array_equal(_sum_val, 10)
