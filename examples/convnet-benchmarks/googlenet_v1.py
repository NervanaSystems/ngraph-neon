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
Convnet-GoogLeNet v1 Benchmark with spelled out neon model framework in one file
https://github.com/soumith/convnet-benchmarks

./googlenet_v1.py

"""

import numpy as np
import neon as ng
import neon.transformers as ngt
from contextlib import closing

from neon.frontend import NeonArgparser, ArrayIterator
from neon.frontend import XavierInit, UniformInit
from neon.frontend import Affine, Convolution, Pooling, Sequential
from neon.frontend import Rectlin, Softmax, GradientDescentMomentum
from neon.frontend import ax
from neon.frontend import make_bound_computation, make_default_callbacks, loop_train  # noqa

np.seterr(all='raise')

parser = NeonArgparser(description=__doc__)
# Default batch_size for convnet-googlenet is 128.
parser.set_defaults(batch_size=128, num_iterations=100)
args = parser.parse_args()

# Setup data provider
image_size = 224
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
bias_init = UniformInit(low=-0.08, high=0.08)


class Inception(Sequential):

    def __init__(self, branch_units, activation=Rectlin(),
                 bias_init=UniformInit(low=-0.08, high=0.08),
                 filter_init=XavierInit()):

        (p1, p2, p3, p4) = branch_units

        self.branch_1 = Convolution((1, 1, p1[0]), activation=activation,
                                    bias_init=bias_init,
                                    filter_init=filter_init)
        self.branch_2 = [Convolution((1, 1, p2[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((3, 3, p2[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=1)]
        self.branch_3 = [Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init),
                         Convolution((5, 5, p3[1]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init, padding=2)]
        self.branch_4 = [Pooling(pool_shape=(3, 3), padding=1, strides=1, pool_type="max"),
                         Convolution((1, 1, p3[0]), activation=activation,
                                     bias_init=bias_init,
                                     filter_init=filter_init)]

    def __call__(self, in_obj):

        branch_1_output = self.branch_1(in_obj)
        branch_2_output = self.branch_2[0](in_obj)
        branch_2_output = self.branch_2[1](branch_2_output)
        branch_3_output = self.branch_3[0](in_obj)
        branch_3_output = self.branch_3[1](branch_3_output)
        branch_4_output = self.branch_4[0](in_obj)
        branch_4_output = self.branch_4[1](branch_4_output)

        outputs = [branch_1_output, branch_2_output, branch_3_output, branch_4_output]
        # This does the equivalent of neon's merge-broadcast
        return ng.concat_along_axis(outputs, branch_1_output.axes.channel_axis())


seq1 = Sequential([Convolution((7, 7, 64), padding=3, strides=2,
                               activation=Rectlin(), bias_init=bias_init,
                               filter_init=XavierInit()),
                   Pooling(pool_shape=(3, 3), padding=1, strides=2, pool_type='max'),
                   Convolution((1, 1, 64), activation=Rectlin(),
                               bias_init=bias_init, filter_init=XavierInit()),
                   Convolution((3, 3, 192), activation=Rectlin(),
                               bias_init=bias_init, filter_init=XavierInit(),
                               padding=1),
                   Pooling(pool_shape=(3, 3), padding=1, strides=2, pool_type='max'),
                   Inception([(64,), (96, 128), (16, 32), (32,)]),
                   Inception([(128,), (128, 192), (32, 96), (64,)]),
                   Pooling(pool_shape=(3, 3), padding=1, strides=2, pool_type='max'),
                   Inception([(192,), (96, 208), (16, 48), (64,)]),
                   Inception([(160,), (112, 224), (24, 64), (64,)]),
                   Inception([(128,), (128, 256), (24, 64), (64,)]),
                   Inception([(112,), (144, 288), (32, 64), (64,)]),
                   Inception([(256,), (160, 320), (32, 128), (128,)]),
                   Pooling(pool_shape=(3, 3), padding=1, strides=2, pool_type='max'),
                   Inception([(256,), (160, 320), (32, 128), (128,)]),
                   Inception([(384,), (192, 384), (48, 128), (128,)]),
                   Pooling(pool_shape=(7, 7), strides=1, pool_type="avg"),
                   Affine(axes=ax.Y, weight_init=XavierInit(),
                          bias_init=bias_init, activation=Softmax())])

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
