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
Usage:
python ./inceptionv3.py -b gpu --mini --optimizer_name rmsprop

Inception v3 network based on:
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
https://arxiv.org/pdf/1512.00567.pdf

Imagenet data needs to be downloaded and extracted from:
http://www.image-net.org/
"""
import numpy as np
import pickle
from tqdm import tqdm
from contextlib import closing
import neon as ng
import neon.transformers as ngt
from neon.frontend import NeonArgparser
from neon.frontend import Layer
from neon.frontend import ax, RMSProp, GradientDescentMomentum
from neon.frontend.model import make_bound_computation
from neon.frontend.callbacks import loop_eval, loop_train, \
    make_default_callbacks
from data import make_aeon_loaders
import inception


parser = NeonArgparser(description=__doc__)
parser.add_argument('--mini', default=False, dest='mini', action='store_true',
                    help='If given, builds a mini version of Inceptionv3')
parser.add_argument("--image_dir", default='/dataset/aeon/I1K/i1k-extracted/',
                    help="Path to extracted imagenet data")
parser.add_argument("--train_manifest_file", default='train-index-tabbed.csv',
                    help="Name of tab separated Aeon training manifest file")
parser.add_argument("--valid_manifest_file", default='val-index-tabbed.csv',
                    help="Name of tab separated Aeon validation manifest file")
parser.add_argument("--optimizer_name", default='sgd',
                    help="Name of optimizer (sgd or rmsprop)")
parser.set_defaults(batch_size=32, num_iterations=10000000, iter_interval=2000)
args = parser.parse_args()

# Set the random seed
np.random.seed(1)
# Number of outputs of last layer.
ax.Y.length = 1000
ax.N.length = args.batch_size

# Build AEON data loader objects
train_set, valid_set = make_aeon_loaders(train_manifest=args.train_manifest_file,
                                         valid_manifest=args.valid_manifest_file,
                                         batch_size=args.batch_size,
                                         train_iterations=args.num_iterations,
                                         dataset='i1k',
                                         datadir=args.image_dir)
inputs = train_set.make_placeholders(include_iteration=True)

# Input size is 299 x 299 x 3
image_size = 299

# Build the network
inception = inception.Inception(mini=args.mini)

# Declare the optimizer
if args.optimizer_name == 'sgd':
    learning_rate_policy = {'name': 'schedule',
                            'schedule': list(7000 * np.arange(1, 10, 1)),
                            'gamma': 0.7,
                            'base_lr': 0.1}

    optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                        momentum_coef=0.5,
                                        wdecay=4e-5,
                                        iteration=inputs['iteration'])
elif args.optimizer_name == 'rmsprop':
    learning_rate_policy = {'name': 'schedule',
                            'schedule': list(80000 * np.arange(1, 10, 1)),
                            'gamma': 0.94,
                            'base_lr': 0.01}
    optimizer = RMSProp(learning_rate=learning_rate_policy,
                        wdecay=4e-5, decay_rate=0.9, momentum_coef=0.9,
                        epsilon=1., iteration=inputs['iteration'])
else:
    raise NotImplementedError("Unrecognized Optimizer")

# Build the main and auxiliary loss functions
y_onehot = ng.one_hot(inputs['label'], axis=ax.Y)
train_prob_prefix = inception.seq1(inputs['image'])
train_prob_main = inception.seq2(train_prob_prefix)
train_loss_main = ng.cross_entropy_multi(train_prob_main, y_onehot)

train_prob_aux = inception.seq_aux(train_prob_prefix)
train_loss_aux = ng.cross_entropy_multi(train_prob_aux, y_onehot)

batch_cost = ng.sequential([optimizer(train_loss_main + 0.4 * train_loss_aux),
                            ng.mean(train_loss_main, out_axes=())])

train_outputs = dict(batch_cost=batch_cost)

# Build the computations for inference (evaluation)
with Layer.inference_mode_on():
    inference_prob = inception.seq2(inception.seq1(inputs['image']))
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
    eval_outputs = dict(results=inference_prob, cross_ent_loss=eval_loss)


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
                                 enable_top5=True,
                                 use_progress_bar=args.progress_bar)

    loop_train(train_set, cbs)
