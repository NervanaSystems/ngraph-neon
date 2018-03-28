#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np
import pytest

import neon as ng
from neon.testing import ExecutorFactory, executor


def test_fill_state():
    with ExecutorFactory() as ex:
        N = ng.make_axis(3, name='N')
        x_np = np.ones((N.length)) * 4
        x = ng.variable([N], initial_value=x_np).named('x')
        val = ng.sequential([
            ng.fill(x, -1),
            x
        ])
        f = ex.executor(val)
        x_val = f()
    assert np.allclose(-1, x_val)

def test_add_with_scalar():

    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=4, name='width')
    a = ng.placeholder(axes=[H, W])

    d = ng.add(a, -5)
    with executor(d, a) as _add:
        d_val = _add([10, 20, 30, 40])

        # compute reference through numpy
        d_val_ref = np.add(np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 4),
                           np.array([-5], dtype=np.float32))

    assert np.allclose(d_val[0], d_val_ref)


def test_add_with_mul():

    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=1, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])
    c = ng.placeholder(axes=[H, W])
    d = ng.multiply(ng.add(a, b), c)

    with executor(d, a, b, c) as _mul:
        d_val = _mul([10], [20], [10])

        # compute reference through numpy
        _add_ref = np.add(np.full([1, 1], 10, dtype=np.float32),
                          np.full([1, 1], 20, dtype=np.float32))
        d_val_ref = np.multiply(_add_ref, np.full([1, 1], 10, dtype=np.float32))
        assert np.allclose(d_val, d_val_ref)


def test_multiple_computation():
    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=1, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])

    _mul = ng.multiply(a, b)
    _add = ng.add(a, b)

    with ExecutorFactory() as ex:
        # Define computations
        _mul_computation = ex.executor(_mul, a, b)
        _mul_val = _mul_computation([10], [20])
        _add_computation = ex.executor(_add, a, b)
        _add_val = _add_computation([10], [20])

        # compute reference value
        _mul_ref = np.multiply(np.full([1, 1], 10, dtype=np.float32),
                               np.full([1, 1], 20, dtype=np.float32))
        _add_ref = np.add(np.full([1, 1], 10, dtype=np.float32),
                          np.full([1, 1], 20, dtype=np.float32))

        assert np.allclose(_add_val, _add_ref)
        assert np.allclose(_mul_val, _mul_ref)


def test_add_with_muliple_axis():
    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=4, name='width')
    a = ng.placeholder(axes=[H, W])
    b = ng.placeholder(axes=[H, W])

    d = ng.add(a, b)

    with executor(d, a, b) as _add:
        d_val = _add([10, 20, 30, 40], [11, 12, 13, 14])

        # compute reference through numpy
        d_val_ref = np.add(np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 4),
                           np.array([11, 12, 13, 14], dtype=np.float32).reshape(1, 4))

        assert np.allclose(d_val, d_val_ref)


def test_broadcast():
    M = ng.make_axis(length=1)
    N = ng.make_axis(length=4)

    np_a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    np_c = np.add(np_a, 2)

    a = ng.constant(np_a, [M, N])
    c = ng.add(a, 2)

    with executor(c) as _add:
        result = _add()

        assert np.allclose(result, np_c)


def test_cast_axis():
    """
    Test AxesCastOp
    """
    H = ng.make_axis(length=1, name='height')
    W = ng.make_axis(length=4, name='width')
    axes_input = [H, W]
    a = ng.placeholder(axes=axes_input)
    axes_output = ng.make_axes([ng.make_axis(name=ax.name + 'p', length=ax.length)
                                for ax in axes_input])

    b = ng.cast_axes(a, axes_output)

    with executor(b, a) as _cast_axis:
        a_val = np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 4)
        b_val = _cast_axis(a_val)

        b_val_ref = a_val
        assert np.allclose(b_val, b_val_ref)


def test_dot():
    H = ng.make_axis(length=1)
    W = ng.make_axis(length=4)
    np_a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    np_b = np.array(3, dtype=np.float32)

    a = ng.constant(np_a, [H, W])
    b = ng.constant(np_b, [])
    c = ng.dot(a, b)

    with executor(c) as _dot:
        _dot_val = _dot()

        # compute reference
        _dot_val_ref = np.dot(np_a, np_b)

        # this checks the dot product between scalar and vector, this is equivalent to
        # elementwise multiplication between scalar and vector
        assert np.allclose(_dot_val, _dot_val_ref)


def test_prod():

    H = ng.make_axis(length=2)
    W = ng.make_axis(length=2)
    H1 = ng.make_axis(length=1)
    W1 = ng.make_axis(length=4)

    input1 = ng.placeholder(axes=[H, W])
    input2 = ng.placeholder(axes=[H1, W1])

    # does reduction sum operation along axis[0]:H
    prod_op_1 = ng.prod(input1, reduction_axes=H)

    # sum elements across all the axis
    prod_op_2 = ng.prod(input2)

    with ExecutorFactory() as ex:
        _prod = ex.executor(prod_op_1, input1)
        _prod_val = _prod([[1, 2], [3, 4]])
        assert np.array_equal(_prod_val, [3, 8])

        _prod = ex.executor(prod_op_2, input2)
        _prod_val = _prod([1, 2, 3, 4])
        assert np.array_equal(_prod_val, 24)


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

    with ExecutorFactory() as ex:
        _sum = ex.executor(sum_op_1, input1)
        _sum_val = _sum([[1, 2], [3, 4]])
        assert np.array_equal(_sum_val, [4, 6])

        _sum = ex.executor(sum_op_2, input2)
        _sum_val = _sum([1, 2, 3, 4])
        assert np.array_equal(_sum_val, 10)


def test_tensor_dot_tensor():
    """TODO."""
    C = ng.make_axis().named('C')
    D = ng.make_axis().named('D')
    H = ng.make_axis().named('H')
    N = ng.make_axis().named('N')

    tests = [
        {
            'tensor1': [[1, 2], [4, 5], [3, 4]],
            'tensor1_axes': (C, D),
            'tensor2': [2, 5],
            'tensor2_axes': (D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {C: 3, D: 2}
        },
        {
            'tensor1': [[1, 4, 3], [2, 5, 4]],
            'tensor1_axes': (D, C),
            'tensor2': [2, 5],
            'tensor2_axes': (D,),
            'expected_output': [12, 33, 26],
            'axes_lengths': {C: 3, D: 2}
        },
        {
            'tensor1': [[[1, 4], [2, 5]], [[7, 12], [13, 2]]],
            'tensor1_axes': (N, D, C),
            'tensor2': [[[3, 6], [7, 2]], [[9, 8], [10, 4]]],
            'tensor2_axes': (H, D, C),
            'expected_output': [[51, 81], [188, 297]],
            'axes_lengths': {N: 2, D: 2, C: 2, H: 2}
        },
        {
            'tensor1': [1, 2],
            'tensor1_axes': (C,),
            'tensor2': [7, 11, 13],
            'tensor2_axes': (D,),
            'expected_output': [[7, 11, 13], [14, 22, 26]],
            'axes_lengths': {C: 2, D: 3}
        },
        {
            'tensor1': [[1, 4], [6, 2]],
            'tensor1_axes': (C, D),
            'tensor2': [[1, 4], [6, 2]],
            'tensor2_axes': (C, D),
            'expected_output': 57,
            'axes_lengths': {C: 2, D: 2}
        }
    ]

    for test in tests:
        # set up axis
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        # set up tensors
        tensor1 = ng.placeholder(test['tensor1_axes'])
        value1 = np.array(test['tensor1'], dtype=np.float32)

        tensor2 = ng.placeholder(test['tensor2_axes'])
        value2 = np.array(
            test['tensor2'], dtype=np.float32
        )

        # compute outputs
        expected_output = np.array(test['expected_output'], dtype=np.float32)

        dot = ng.dot(tensor1, tensor2)

        with executor(dot, tensor1, tensor2) as evaluated_fun:
            # assert outputs are equal
            evaluated = evaluated_fun(value1, value2)
            np.testing.assert_equal(evaluated, expected_output)


@pytest.mark.parametrize('ng_func, np_func', [
    (ng.Maximum, np.maximum),
    (ng.Minimum, np.minimum),
    (ng.Greater, np.greater),
    (ng.GreaterEqual, np.greater_equal),
    (ng.Equal, np.equal),
    (ng.NotEqual, np.not_equal),
    (ng.Less, np.less),
    (ng.LessEqual, np.less_equal),
    (ng.power, np.power)
])
def test_binary_op(ng_func, np_func):
    H = ng.make_axis().named('H')
    W = ng.make_axis().named('W')

    tests = [
        {
            'tensor1': [[1, 2, 3, 4], [8, 7, 6, 5]],
            'tensor1_axes': (H, W),
            'tensor2': [[10, 2, 3, 40], [15, 6, 9, 8]],
            'tensor2_axes': (H, W),
            'axes_lengths': {H: 2, W: 4}
        }]

    for test in tests:
        # set up tensors
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        tensor1 = ng.placeholder(test['tensor1_axes'])
        value1 = np.array(test['tensor1'], dtype=np.float32)

        tensor2 = ng.placeholder(test['tensor2_axes'])
        value2 = np.array(
            test['tensor2'], dtype=np.float32
        )

        _ng_func = ng_func(tensor1, tensor2)

        with ExecutorFactory() as ex:

            _ng_computation = ex.executor(_ng_func, tensor1, tensor2)
            _ng_val = _ng_computation(value1, value2)
            _ng_ref = np_func(value1, value2)
            np.testing.assert_equal(_ng_val, _ng_ref)


@pytest.mark.parametrize('ng_func, np_func', [
    (ng.exp, np.exp),
    (ng.NegativeOp, np.negative),
    (ng.log, np.log),
    (ng.tanh, np.tanh),
    (ng.square, np.square),
    (ng.sqrt, np.sqrt)
])
def test_unary_op_(ng_func, np_func):
    H = ng.make_axis().named('H')
    W = ng.make_axis().named('W')

    tests = [
        {
            'tensor1': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'tensor1_axes': (H, W),
            'axes_lengths': {H: 2, W: 4}
        }]

    for test in tests:
        # set up tensors
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        tensor1 = ng.placeholder(test['tensor1_axes'])
        value1 = np.array(test['tensor1'], dtype=np.float32)

        _ng_func = ng_func(tensor1)

        with ExecutorFactory() as ex:
            _ng_computation = ex.executor(_ng_func, tensor1)
            _ng_val = _ng_computation(value1)
            _ng_ref = np_func(value1)
            assert np.allclose(_ng_val, _ng_ref, rtol=0, atol=2)
