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
"""
LeNet model on MNIST dataset.
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

"""

from __future__ import division
from __future__ import print_function
from contextlib import closing
import numpy as np
import neon as ng
from neon.frontend import Layer, Affine, Preprocess, Convolution, Pooling, Sequential
from neon.frontend import XavierInit, Rectlin, Softmax, GradientDescentMomentum
from neon.frontend import ax, loop_train
from neon.frontend import NeonArgparser, make_bound_computation, make_default_callbacks
from neon.frontend import ArrayIterator

from neon.frontend import MNIST
import neon.transformers as ngt

parser = NeonArgparser(description='Train LeNet topology on Mnist dataset')
args = parser.parse_args()

np.random.seed(args.rng_seed)

# Create the dataloader
train_data, valid_data = MNIST(args.data_dir).load_data()
train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
valid_set = ArrayIterator(valid_data, args.batch_size)

inputs = train_set.make_placeholders()
ax.Y.length = 10

######################
# Model specification

init_xav = XavierInit()

seq1 = Sequential([Preprocess(functor=lambda x: x / 255.),
                   Convolution((5, 5, 16), filter_init=init_xav, activation=Rectlin()),
                   Pooling((2, 2), strides=2),
                   Convolution((5, 5, 32), filter_init=init_xav, activation=Rectlin()),
                   Pooling((2, 2), strides=2),
                   Affine(nout=500, weight_init=init_xav, activation=Rectlin()),
                   Affine(axes=ax.Y, weight_init=init_xav, activation=Softmax())])

optimizer = GradientDescentMomentum(0.01, 0.9)
train_prob = seq1(inputs['image'])
train_loss = ng.cross_entropy_binary(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

with Layer.inference_mode_on():
    inference_prob = seq1(inputs['image'])
eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
eval_outputs = dict(cross_ent_loss=eval_loss, results=inference_prob)

# Now bind the computations we are interested in
with closing(ngt.make_transformer()) as transformer:
    train_computation = make_bound_computation(transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(transformer, eval_outputs, inputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=valid_set,
                                 loss_computation=loss_computation,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, cbs)
