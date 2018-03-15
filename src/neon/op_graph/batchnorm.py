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


def batchnormtrain(inputs, gamma, beta, epsilon, out_axes, axes, docstring=None):
    """

    Args:
        inputs (TensorOp): The input tensor.
        gamma (TensorOp): Gamma for batchnorm.
        beta (TensorOp): Beta for batchnorm
        epsilon (TensorOp): Epsilon for batchnorm
        docstring (String, optional): Documentation for the op.

    Returns:
        Tuple of TensorOp: Result of the batchnorm (Output, mean and variance)
    """
    bnc = batchnormcommon(inputs, gamma, beta, epsilon, axes)
    bnoutput = batchnormoutput(bnc, axes)
    xmean = batchnormmean(bnc, out_axes)
    bnc.lmean = xmean
    xvar = batchnormvar(bnc, out_axes)
    bnc.lvar = xvar
    return (bnoutput, xmean, xvar)


def batchnormcommon(inputs, gamma, beta, epsilon, axes, docstring=None):
    """

    Args:
        inputs (TensorOp): The input tensor.
        gamma (TensorOp): Gamma for batchnorm.
        beta (TensorOp): Beta for batchnorm
        epsilon (TensorOp): Epsilon for batchnorm
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: An op that holds result of the batchnorm (Output, mean and variance)
    """
    return BatchnormCommonOp(inputs, gamma, beta, epsilon, axes=axes, docstring=docstring)


class BatchnormCommonOp(TensorOp):
    def __init__(self, inputs, gamma, beta, epsilon, **kwargs):
        super(BatchnormCommonOp, self).__init__(
            args=(inputs, gamma, beta), **kwargs)
        self.lmean = None
        self.lvar = None
        self.epsilon = epsilon

    def generate_adjoints(self, adjoints, delta, inputs, gamma, beta):
        if (self.lmean is None) or (self.lvar) is None:
            RuntimeError("mean and variance op must be set for Batchnorm autodiff")
        bnbc = batchnormbpropcommon(
            inputs, gamma, beta, self.lmean, self.lvar, delta, self.epsilon, self.axes)
        bprop_data_op = batchnormbpropdata(bnbc, axes=inputs.axes)
        bprop_gamma_op = batchnormbpropgamma(bnbc, axes=gamma.axes)
        bprop_beta_op = batchnormbpropbeta(bnbc, axes=beta.axes)
        inputs.generate_add_delta(adjoints, bprop_data_op)
        gamma.generate_add_delta(adjoints, bprop_gamma_op)
        beta.generate_add_delta(adjoints, bprop_beta_op)


def batchnormoutput(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormCommonOp): The input tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Output of the batchnorm.
    """
    return BatchnormOutputOp(inputs, axes=axes, docstring=docstring)


class BatchnormOutputOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormOutputOp, self).__init__(args=(inputs,), **kwargs)

    def generate_adjoints(self, adjoints, delta, inputs):
        inputs.generate_add_delta(adjoints, delta)


def batchnormmean(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormCommonOp): The input tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Mean from the batchnorm.
    """
    return BatchnormMeanOp(inputs, axes=axes, docstring=docstring)


class BatchnormMeanOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormMeanOp, self).__init__(args=(inputs,), **kwargs)
        inputs.lmean = self


def batchnormvar(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormCommonOp): The input tensor.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Variance from the batchnorm.
    """
    return BatchnormVarOp(inputs, axes=axes, docstring=docstring)


class BatchnormVarOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormVarOp, self).__init__(args=(inputs,), **kwargs)


def batchnormbpropcommon(inputs, gamma, beta, mean, variance,
                         delta, epsilon, axes, docstring=None):
    """

    Args:
        inputs (TensorOp): The input tensor for batchnorm.
        gamma (TensorOp): Gamma for batchnorm.
        beta (TensorOp): Beta for batchnorm
        mean (TensorOp): Mean from batchnorm
        variance (TensorOp): Variance from batchnorm
        delta (TensorOp): Delta for bprop
        epsilon (TensorOp): Epsilon for batchnorm
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: An op that holds result of the batchnorm bprop (delta to inputs, gamma and beta)
    """
    return BatchnormBpropCommonOp(
        inputs, gamma, beta, mean, variance, delta, epsilon, axes=axes, docstring=docstring)


class BatchnormBpropCommonOp(TensorOp):
    def __init__(self, inputs, gamma, beta, mean, variance, delta, epsilon, **kwargs):
        super(BatchnormBpropCommonOp, self).__init__(
            args=(inputs, gamma, beta, mean, variance, delta), **kwargs)
        self.epsilon = epsilon


def batchnormbpropdata(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormBpropCommonOp): Deltas from batchnorm bprop.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Delta from the batchnorm to batchnorm inputs.
    """
    return BatchnormBpropDataOp(inputs, axes=axes, docstring=docstring)


class BatchnormBpropDataOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormBpropDataOp, self).__init__(args=(inputs,), **kwargs)


def batchnormbpropgamma(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormBpropCommonOp): Deltas from batchnorm bprop.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Delta from the batchnorm to batchnorm gamma.
    """
    return BatchnormBpropGammaOp(inputs, axes=axes, docstring=docstring)


class BatchnormBpropGammaOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormBpropGammaOp, self).__init__(args=(inputs,), **kwargs)


def batchnormbpropbeta(inputs, axes, docstring=None):
    """

    Args:
        inputs (BatchnormBpropCommonOp): Deltas from batchnorm bprop.
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: Delta from the batchnorm to batchnorm beta.
    """
    return BatchnormBpropBetaOp(inputs, axes=axes, docstring=docstring)


class BatchnormBpropBetaOp(TensorOp):
    def __init__(self, inputs, **kwargs):
        super(BatchnormBpropBetaOp, self).__init__(args=(inputs,), **kwargs)


def batchnorminference(inputs, gamma, beta, mean, variance, epsilon, axes, docstring=None):
    """

    Args:
        inputs (TensorOp): The input tensor.
        gamma (TensorOp): Gamma for batchnorm.
        beta (TensorOp): Beta for batchnorm
        mean (TensorOp): Beta for batchnorm
        variance (TensorOp): Mean for batchnorm
        epsilon (TensorOp): Variance for batchnorm
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: The result of the batchnorm for inference.
    """
    return BatchnormInferenceOp(
        inputs, gamma, beta, mean, variance, epsilon, axes=axes, docstring=docstring)


class BatchnormInferenceOp(TensorOp):
    def __init__(self, inputs, gamma, beta, mean, variance, epsilon, **kwargs):
        super(BatchnormInferenceOp, self).__init__(
            args=(inputs, gamma, beta, mean, variance), **kwargs)
        self.epsilon = epsilon
