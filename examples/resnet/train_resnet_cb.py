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

from __future__ import division, print_function
import sys
import numpy as np
import neon as ng
import neon.transformers as ngt
from neon.frontend import ax, NeonArgparser
from tqdm import tqdm
from data import make_aeon_loaders
from neon.frontend import GradientDescentMomentum
from neon.frontend import Layer
from resnet import BuildResnet
from contextlib import closing
from neon.frontend import Saver
from neon.frontend import FeedAddWrapper
from neon.frontend.model import make_bound_computation
from neon.frontend.callbacks import loop_eval, loop_train, \
    make_default_callbacks, TrainSaverCallback
from utils import get_network_params, set_lr

if __name__ == "__main__":
    # Hyperparameters
    # Optimizer
    base_lr = 0.1
    gamma = 0.1
    momentum_coef = 0.9
    wdecay = 0.0001
    nesterov = False

    print("HyperParameters")
    print("Learning Rate:     " + str(base_lr))
    print("Momentum:          " + str(momentum_coef))
    print("Weight Decay:      " + str(wdecay))
    print("Nesterov:          " + str(nesterov))

    # Command Line Parser
    parser = NeonArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('--dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
    parser.add_argument('--tb', action="store_true", help="1- Enables tensorboard")
    parser.add_argument('--disable_batch_norm', action='store_true')
    parser.add_argument('--save_file', type=str, default=None, help="File to save weights")
    parser.add_argument('--inference', type=str, default=None, help="File to load weights")
    parser.add_argument('--resume', type=str, default=None, help="Weights file to resume training")
    args = parser.parse_args()

# Initialize seed before any use
np.random.seed(args.rng_seed)

# Get network parameters
nw_params = get_network_params(args.dataset, args.size, args.batch_size)
metric_names = nw_params['metric_names']
en_top5 = nw_params['en_top5']
num_resnet_mods = nw_params['num_resnet_mods']
args.iter_interval = nw_params['iter_interval']
learning_schedule = nw_params['learning_schedule']
en_bottleneck = nw_params['en_bottleneck']
ax.Y.length = nw_params['num_classes']

# Set batch size
ax.N.length = args.batch_size

# Create training and validation set objects
train_set, valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                         args.num_iterations, dataset=args.dataset)
print("Completed loading " + args.dataset + " dataset")
# Randomize seed
np.random.seed(args.rng_seed)
# Make placeholders
input_ph = train_set.make_placeholders(include_iteration=True)

resnet = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods,
                     batch_norm=not args.disable_batch_norm)

# Tensorboard
if(args.tb):
    try:
        from neon.op_graph.tensorboard.tensorboard import TensorBoard
    except:
        print("Tensorboard not installed")
    seq1 = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods)
    train = seq1(input_ph['image'])
    tb = TensorBoard("/tmp/")
    tb.add_graph(train)
    exit()

# Learning Rate Placeholder
lr_ph = ng.placeholder(axes=(), initial_value=base_lr)

# Optimizer
# Provided learning policy takes learning rate as input to graph using a placeholder.
# This allows you to control learning rate based on various factors of network
learning_rate_policy = {'name': 'provided',
                        'lr_placeholder': lr_ph}

optimizer = GradientDescentMomentum(learning_rate=learning_rate_policy,
                                    momentum_coef=momentum_coef,
                                    wdecay=wdecay,
                                    nesterov=False,
                                    iteration=input_ph['iteration'])
label_indices = input_ph['label']
# Make a prediction
prediction = resnet(input_ph['image'])
# Calculate loss
train_loss = ng.cross_entropy_multi(prediction, ng.one_hot(label_indices, axis=ax.Y))
# Average loss over the batch
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_outputs = dict(batch_cost=batch_cost)

# Instantiate the Saver object to save weights
weight_saver = Saver()

with Layer.inference_mode_on():
    # Doing inference
    inference_prob = resnet(input_ph['image'])
    eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
    # Computation for inference
    eval_outputs = dict(results=inference_prob, cross_ent_loss=eval_loss)

# setup wrapper for additional feed for learning rate (train only)
input_ph['lr_ph'] = lr_ph
wrapper_kwargs = {'base_lr': base_lr,
                  'learning_schedule': learning_schedule,
                  'gamma': gamma}
train_feed_wrapper = FeedAddWrapper(wrapper=set_lr,
                                    holder='lr_ph',
                                    wrapper_kwargs=wrapper_kwargs)

# Doing inference
if(args.inference is not None):
    # Check if file exists. TODO.
    with closing(ngt.make_transformer()) as transformer:
        restore_loss_computation = make_bound_computation(transformer,
                                                          eval_outputs,
                                                          input_ph)
        weight_saver.setup_restore(transformer=transformer, computation=eval_outputs,
                                   filename=args.inference)
        # Restore weight
        weight_saver.restore()
        # Calculate losses

        eval_losses = loop_eval(valid_set,
                                restore_loss_computation,
                                enable_top5=en_top5)
        # Print statistics
        print("From restored weights: Test Avg loss:{tcost}".format(tcost=eval_losses))
        exit()

# Training the network by calling transformer
with closing(ngt.make_transformer()) as transformer:
    # Trainer
    train_computation = make_bound_computation(transformer, train_outputs, input_ph)
    # Inference
    loss_computation = make_bound_computation(transformer, eval_outputs, input_ph)
    # Set Saver for saving weights
    weight_saver.setup_save(transformer=transformer, computation=train_outputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=valid_set,
                                 loss_computation=loss_computation,
                                 enable_top5=en_top5,
                                 use_progress_bar=args.progress_bar)
    if(args.save_file is not None):
        cbs.append(TrainSaverCallback(saver=weight_saver,
                                      filename=args.save_file,
                                      frequency=args.iter_interval))
    loop_train(train_set, cbs, train_feed_wrapper=train_feed_wrapper)

    print("\nTraining Completed")
    if(args.save_file is not None):
        print("\nSaving Weights")
        weight_saver.save(filename=args.save_file)
