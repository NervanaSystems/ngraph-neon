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
"""
CIFAR MSRA with spelled out neon model framework in one file

Run it using

python examples/cifar10/cifar10_msra.py --stage_depth 2 --data_dir /usr/local/data/CIFAR

For full training, the number of iterations should be 64000 with batch size 128.

"""
from __future__ import division, print_function
from builtins import range
from contextlib import closing
import numpy as np
import neon as ng
from neon.frontend import Layer, Sequential
from neon.frontend import Affine, Preprocess, Convolution, Pooling, BatchNorm, Activation
from neon.frontend import KaimingInit, Rectlin, Softmax, GradientDescentMomentum
from neon.frontend import ax, NeonArgparser
from neon.frontend import make_bound_computation, make_default_callbacks, loop_train  # noqa
from tqdm import tqdm
import neon.transformers as ngt

######################
# Model specification


def cifar_mean_subtract(x):
    # Assign roles
    bgr_mean = ng.persistent_tensor(
        axes=[x.axes.channel_axis()],
        initial_value=np.array([104., 119., 127.]))

    return (x - bgr_mean) / 255.


def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(filter_shape=(fsize, fsize, nfm),
                strides=strides,
                padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                filter_init=KaimingInit(),
                batch_norm=batch_norm)


class f_module(object):

    def __init__(self, nfm, first=False, strides=1):

        self.trunk = None
        self.side_path = None

        main_path = [Convolution(**conv_params(1, nfm, strides=strides)),
                     Convolution(**conv_params(3, nfm)),
                     Convolution(**conv_params(1, nfm * 4, relu=False, batch_norm=False))]

        if first or strides == 2:
            self.side_path = Convolution(
                **conv_params(1, nfm * 4, strides=strides, relu=False, batch_norm=False))
        else:
            main_path = [BatchNorm(), Activation(Rectlin())] + main_path

        if strides == 2:
            self.trunk = Sequential([BatchNorm(), Activation(Rectlin())])

        self.main_path = Sequential(main_path)

    def __call__(self, x):
        t_x = self.trunk(x) if self.trunk else x
        s_y = self.side_path(t_x) if self.side_path else t_x
        m_y = self.main_path(t_x)
        return s_y + m_y


class residual_network(Sequential):

    def __init__(self, stage_depth):
        nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * stage_depth)]
        print(nfms)
        strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

        layers = [Preprocess(functor=cifar_mean_subtract),
                  Convolution(**conv_params(3, 16)),
                  f_module(nfms[0], first=True)]

        for nfm, stride in zip(nfms[1:], strides):
            layers.append(f_module(nfm, strides=stride))

        layers.append(BatchNorm())
        layers.append(Activation(Rectlin()))
        layers.append(Pooling((8, 8), pool_type='avg'))
        layers.append(Affine(axes=ax.Y,
                             weight_init=KaimingInit(),
                             activation=Softmax()))
        super(residual_network, self).__init__(layers=layers)


if __name__ == "__main__":
    parser = NeonArgparser(description='Train deep residual network on cifar10 dataset')
    parser.add_argument('--stage_depth', type=int, default=2,
                        help='depth of each stage (network depth will be 9n+2)')
    parser.add_argument('--use_aeon', action='store_true', help='whether to use aeon dataloader')
    args = parser.parse_args()

    np.random.seed(args.rng_seed)

    # Create the dataloader
    if args.use_aeon:
        from data import make_aeon_loaders
        train_set, valid_set = make_aeon_loaders(args.data_dir,
                                                 args.batch_size,
                                                 args.num_iterations)
    else:
        from neon.frontend import ArrayIterator  # noqa
        from neon.frontend import CIFAR10  # noqa
        train_data, valid_data = CIFAR10(args.data_dir).load_data()
        train_set = ArrayIterator(train_data, args.batch_size,
                                  total_iterations=args.num_iterations)
        valid_set = ArrayIterator(valid_data, args.batch_size)

    # we need to ask the dataset to create an iteration
    # placeholder for our learning rate schedule
    inputs = train_set.make_placeholders(include_iteration=True)
    ax.Y.length = 10

    resnet = residual_network(args.stage_depth)

    learning_rate_policy = {'name': 'schedule',
                            'schedule': [32000, 48000],
                            'gamma': 0.1,
                            'base_lr': 0.1}

    optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                        momentum_coef=0.9,
                                        wdecay=0.0001,
                                        iteration=inputs['iteration'])
    label_indices = inputs['label']
    train_loss = ng.cross_entropy_multi(resnet(inputs['image']),
                                        ng.one_hot(label_indices, axis=ax.Y))
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_outputs = dict(batch_cost=batch_cost)

    with Layer.inference_mode_on():
        inference_prob = resnet(inputs['image'])
        eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
        eval_outputs = dict(results=inference_prob, cross_ent_loss=eval_loss)

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
                                     enable_top5=True,
                                     use_progress_bar=args.progress_bar)

        loop_train(train_set, cbs)
