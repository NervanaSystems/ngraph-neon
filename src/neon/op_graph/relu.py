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
from __future__ import division
from neon.op_graph.op_graph import TensorOp


def relu(inputs, axes, docstring=None):
    """
    Args:
        inputs (TensorOp): The input tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: The result of Relu
    """
    return ReluOp(inputs, axes=axes, docstring=docstring)


class ReluOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(ReluOp, self).__init__(args=(inputs,), **kwargs)

    def generate_adjoints(self, adjoints, delta, inputs):
        relu_bprop = ReluBpropOp(inputs, delta, axes=inputs.axes)
        inputs.generate_add_delta(adjoints, relu_bprop)


class ReluBpropOp(TensorOp):
    def __init__(self, inputs, delta, **kwargs):
        super(ReluBpropOp, self).__init__(args=(inputs, delta), **kwargs)
