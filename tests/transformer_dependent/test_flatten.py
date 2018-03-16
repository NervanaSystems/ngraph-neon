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

from neon.testing import check_derivative, RandomTensorGenerator
import numpy as np
import pytest
import neon as ng
rng = RandomTensorGenerator(0, np.float32)

pytestmark = pytest.mark.transformer_dependent


def test_flatten_deriv_simplified():
    """
    Test derivative with dot and flatten
    """
    ax_N = ng.make_axis(length=3)
    ax_Y = ng.make_axis(length=2)

    x = ng.placeholder(ng.make_axes([ax_N]))
    w = ng.constant([5, 2], axes=ng.make_axes([ax_Y]))
    logits = ng.dot(x, w)
    cost = ng.sum(logits, reduction_axes=logits.axes)

    delta = 0.001
    u = rng.uniform(.1, 5.0, x.axes)
    check_derivative(cost, x, delta, u, atol=1e-2, rtol=1e-2)


