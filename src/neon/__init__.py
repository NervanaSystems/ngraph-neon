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

from __future__ import print_function

import os
import json
import logging
import logging.config
import neon.transformers as transformers
from neon.op_graph.axes import make_axis, make_axes
from neon.transformers.base import UnsupportedTransformerException

from neon.op_graph.convolution import convolution, deconvolution
from neon.op_graph.pooling import pooling
from neon.op_graph.lookuptable import lookuptable
from neon.op_graph.batchnorm import batchnormcommon, batchnorminference, \
    batchnormoutput, batchnormmean, batchnormvar, batchnormbpropcommon, \
    batchnormbpropdata, batchnormbpropgamma, batchnormbpropbeta, batchnormtrain
from neon.op_graph.relu import relu
from neon.op_graph.debug import PrintOp
from neon.op_graph.op_graph import *
from neon.op_graph.op_graph import axes_with_order, \
    broadcast, cast_axes, \
    persistent_tensor, placeholder, \
    slice_along_axis, temporary, \
    add, as_op, as_ops, constant, variable, persistent_tensor, placeholder, \
    temporary, variance, squared_L2, \
    negative, absolute, sin, cos, tanh, exp, log, reciprocal, safelog, sign, \
    square, sqrt, tensor_size, assign, batch_size, pad, sigmoid, \
    one_hot, stack
import neon.testing as testing

__all__ = [
    'absolute',
    'add',
    'as_op',
    'as_ops',
    'axes_with_order',
    'batch_size',
    'broadcast',
    'cast_axes',
    'computation',
    'constant',
    'convolution',
    'cos',
    'deconvolution',
    'exp',
    'fill',
    'log',
    'lookuptable',
    'make_axes',
    'make_axis',
    'negative',
    'one_hot',
    'pad',
    'persistent_tensor',
    'placeholder',
    'pooling',
    'reciprocal',
    'safelog',
    'sequential',
    'sigmoid',
    'sign',
    'sin',
    'slice_along_axis',
    'sqrt',
    'square',
    'squared_L2',
    'stack',
    'tanh',
    'temporary',
    'testing',
    'tensor_size',
    'tensor_slice',
    'value_of',
    'variable',
    'variance',
]

# Set default logging behavior to avoid "No handler found" warnings. And provide sane defaults.
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.json')
logging.config.dictConfig(json.load(open(config_path)))
if os.environ.get('NEON_LOG', None) in ('ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'):
    for handler in logging.getLogger().handlers:
        lvl = getattr(logging, os.environ['NEON_LOG'])
        handler.setLevel(lvl)

# Optionally we can act like a 'good library citizen' and not have any defaults, forcing the user
# to set everything up:
# logging.getLogger(__name__).addHandler(NullHandler())

