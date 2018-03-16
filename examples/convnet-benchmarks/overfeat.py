#!/usr/bin/env python
# ******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
"""
Convnet-overfeat Benchmark with spelled out neon model framework in one file
https://github.com/soumith/convnet-benchmarks

./overfeat.py

"""

import numpy as np
import neon as ng
import neon.transformers as ngt
from contextlib import closing

from neon.frontend import NeonArgparser, ArrayIterator
from neon.frontend import GaussianInit, UniformInit
from neon.frontend import Affine, Convolution, Pooling, Sequential
from neon.frontend import Rectlin, Softmax, GradientDescentMomentum
from neon.frontend import ax
from neon.frontend import make_bound_computation, make_default_callbacks, loop_train  # noqa

np.seterr(all='raise')

parser = NeonArgparser(description='Train convnet-overfeat model on random dataset')
# Default batch_size for convnet-overfeat is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 231
X_train = np.random.uniform(-1, 1, (args.batch_size, 3, image_size, image_size))
y_train = np.ones(shape=(args.batch_size), dtype=np.int32)
train_data = {'image': {'data': X_train,
                        'axes': ('N', 'C', 'H', 'W')},
              'label': {'data': y_train,
                        'axes': ('N',)}}
train_set = ArrayIterator(train_data,
                          batch_size=args.batch_size,
                          total_iterations=args.num_iterations)
inputs = train_set.make_placeholders(include_iteration=True)
ax.Y.length = 1000  # number of outputs of last layer.

# weight initialization
init = UniformInit(low=-0.08, high=0.08)

# Setup model
seq1 = Sequential([Convolution((11, 11, 96), filter_init=GaussianInit(std=0.01),
                               bias_init=init,
                               activation=Rectlin(), padding=0, strides=4),
                   Pooling((2, 2), strides=2),
                   Convolution((5, 5, 256), filter_init=GaussianInit(std=0.01),
                               bias_init=init,
                               activation=Rectlin(), padding=0),
                   Pooling((2, 2), strides=2),
                   Convolution((3, 3, 512), filter_init=GaussianInit(std=0.01),
                               bias_init=init,
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 1024), filter_init=GaussianInit(std=0.01),
                               bias_init=init,
                               activation=Rectlin(), padding=1),
                   Convolution((3, 3, 1024), filter_init=GaussianInit(std=0.01),
                               bias_init=init,
                               activation=Rectlin(), padding=1),
                   Pooling((2, 2), strides=2),
                   Affine(nout=3072, weight_init=GaussianInit(std=0.01),
                          bias_init=init,
                          activation=Rectlin()),
                   Affine(nout=4096, weight_init=GaussianInit(std=0.01),
                          bias_init=init,
                          activation=Rectlin()),
                   Affine(axes=ax.Y, weight_init=GaussianInit(std=0.01),
                          bias_init=init,
                          activation=Softmax())])

# Learning rate change based on schedule from learning_rate_policies.py
lr_schedule = {'name': 'schedule', 'base_lr': 0.01,
               'gamma': (1 / 250.)**(1 / 3.),
               'schedule': [22, 44, 65]}
optimizer = GradientDescentMomentum(lr_schedule, 0.0, wdecay=0.0005,
                                    iteration=inputs['iteration'])
train_prob = seq1(inputs['image'])
train_loss = ng.cross_entropy_multi(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, cbs)
