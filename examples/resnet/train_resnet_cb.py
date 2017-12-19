#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# HETR Usage:
# - set environment variables for HETR_AEON_IP and HETR_AEON_PORT
# - run the aeon-service on antoher linux terminal
#   - cd <PATH_TO_AEON_SERVICE>
#   - ./aeon-service --uri http://<HETR_AEON_IP>:<HETR_AEON_PORT>
# - launch train_resnet.py with the desired arg parameters
# ----------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.frontends.neon import ax, NgraphArgparser
from tqdm import tqdm
from data import make_aeon_loaders
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import loop_eval, loop_train
from ngraph.frontends.neon import make_default_callbacks, make_bound_computation
from resnet import BuildResnet
from contextlib import closing
from ngraph.frontends.neon import TrainSaverCallback
from ngraph.frontends.neon import Saver, FeedAddWrapper
from utils import get_network_params, set_lr
import os
import logging

logger = logging.getLogger(__name__)


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
    parser = NgraphArgparser(description="Resnet for Imagenet and Cifar10")
    parser.add_argument('--dataset', type=str, default="cifar10", help="Enter cifar10 or i1k")
    parser.add_argument('--size', type=int, default=56, help="Enter size of resnet")
    parser.add_argument('--tb', action="store_true", help="1- Enables tensorboard")
    parser.add_argument('--logfile', type=str, default=None, help="Name of the csv which \
                        logs different metrics of model")
    parser.add_argument('--hetr_device', type=str, default='cpu', help="hetr device type")
    parser.add_argument('--num_devices', '-m', type=int, default=1, help="num hetr devices")
    parser.add_argument('--disable_batch_norm', action='store_true')
    parser.add_argument('--save_file', type=str, default=None, help="File to save weights")
    parser.add_argument('--inference', type=str, default=None, help="File to load weights")
    args = parser.parse_args()

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

# Initialize device
device_backend = args.backend
device_hetr = args.hetr_device
device_id = [str(d) for d in range(args.num_devices)]

aeon_adress = None
aeon_port = None
if device_backend == 'hetr' and args.num_devices > 1:
    if 'HETR_AEON_IP' not in os.environ or 'HETR_AEON_PORT' not in os.environ:
        raise ValueError('To run hetr with more than one device, you need to set an ip address \
            for the HETR_AEON_IP environment variable and a port number for the HETR_AEON_PORT \
            environment variable')

    aeon_adress = os.environ['HETR_AEON_IP']
    aeon_port = int(os.environ['HETR_AEON_PORT'])

# Create training and validation set objects
# two validation dataloaders sets are created,
# one for the input_ops to fetch the data from aeon on the worker process
# the other to get evalutation stats on the result on the hetr master process
train_set, valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                         args.num_iterations, dataset=args.dataset,
                                         num_devices=args.num_devices, device=device_backend,
                                         split_batch=True, address=aeon_adress, port=aeon_port)
master_valid_set = make_aeon_loaders(args.data_dir, args.batch_size,
                                     args.num_iterations, dataset=args.dataset,
                                     num_devices=1, device=device_backend,
                                     split_batch=False, address=None, port=None,
                                     return_train=False, return_valid=True)
print("Completed loading " + args.dataset + " dataset")
# Randomize seed
np.random.seed(args.rng_seed)

# Make input_ops or placeholders depending on single device or multi device compute
input_ops_train = train_set.make_input_ops("train", aeon_adress, aeon_port, ax.N,
                                           device_hetr, device_id)
input_ops_valid = valid_set.make_input_ops("valid", aeon_adress, aeon_port, ax.N,
                                           device_hetr, device_id)

with ng.metadata(device=device_hetr, device_id=device_id, parallel=ax.N):
    # Build the network
    resnet = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods,
                         batch_norm=not args.disable_batch_norm)

    # Tensorboard
    if(args.tb):
        try:
            from ngraph.op_graph.tensorboard.tensorboard import TensorBoard
        except:
            print("Tensorboard not installed")
        seq1 = BuildResnet(args.dataset, args.size, en_bottleneck, num_resnet_mods)
        train = seq1(input_ops_train['image'])
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
                                        iteration=input_ops_train['iteration'])
    # Make a prediction
    prediction = resnet(input_ops_train['image'])
    # Calculate loss
    train_loss = ng.cross_entropy_multi(prediction,
                                        ng.one_hot(input_ops_train['label'], axis=ax.Y))
    # Average loss over the batch
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_outputs = dict(batch_cost=batch_cost)

# Instantiate the Saver object to save weights
weight_saver = Saver()

with ng.metadata(device=device_hetr, device_id=device_id, parallel=ax.N):
    # Inference
    with Layer.inference_mode_on():
        # Doing inference
        inference_prob = resnet(input_ops_valid['image'])
        eval_loss = ng.cross_entropy_multi(inference_prob,
                                           ng.one_hot(input_ops_valid['label'], axis=ax.Y))
        # Computation for inference
        eval_outputs = dict(results=inference_prob, cross_ent_loss=eval_loss)

# setup wrapper for additional feed for learning rate (train only)
input_ops_train['lr_ph'] = lr_ph
wrapper_kwargs = {'base_lr': base_lr,
                  'learning_schedule': learning_schedule,
                  'gamma': gamma}
lr_add_wrapper = FeedAddWrapper(wrapper=set_lr,
                                holder='lr_ph',
                                wrapper_kwargs=wrapper_kwargs)

# setup feed modifier for HeTr
clear_wrapper = FeedAddWrapper(clear_feed=True)

if device_backend == 'hetr' and args.num_devices > 1:
    train_feed_wrapper = clear_wrapper
    loss_feed_wrapper = clear_wrapper
else:
    train_feed_wrapper = lr_add_wrapper
    loss_feed_wrapper = None

# Doing inference
if(args.inference is not None):
    # Check if file exists. TODO.
    with closing(ngt.make_transformer()) as transformer:
        restore_loss_computation = make_bound_computation(transformer,
                                                          eval_outputs,
                                                          input_ops_valid)
        weight_saver.setup_restore(transformer=transformer, computation=eval_outputs,
                                   filename=args.inference)
        # Restore weight
        weight_saver.restore()
        # Calculate losses

        eval_losses = loop_eval(master_valid_set,
                                restore_loss_computation,
                                en_top5,
                                eval_feed_wrapper=loss_feed_wrapper)
        # Print statistics
        print("From restored weights: Test Avg loss:{tcost}".format(tcost=eval_losses))
        exit()

# Training the network by calling transformer
t_args = {'device': args.hetr_device} if args.backend == 'hetr' else {}
with closing(ngt.make_transformer_factory(args.backend, **t_args)()) as transformer:
    # Trainer
    train_computation = make_bound_computation(transformer, train_outputs, input_ops_train)
    # Inference
    loss_computation = make_bound_computation(transformer, eval_outputs, input_ops_valid)
    # Set Saver for saving weights
    weight_saver.setup_save(transformer=transformer, computation=train_outputs)

    cbs = make_default_callbacks(transformer=transformer,
                                 output_file=args.output_file,
                                 frequency=args.iter_interval,
                                 train_computation=train_computation,
                                 total_iterations=args.num_iterations,
                                 eval_set=master_valid_set,
                                 eval_feed_wrapper=loss_feed_wrapper,
                                 loss_computation=loss_computation,
                                 enable_top5=False,
                                 use_progress_bar=args.progress_bar)
    if(args.save_file is not None):
        cbs.append(TrainSaverCallback(saver=weight_saver,
                                      filename=args.save_file,
                                      frequency=args.iter_interval))
    loop_train(train_set, train_computation, cbs, train_feed_wrapper=train_feed_wrapper)

    print("\nTraining Completed")
    if(args.save_file is not None):
        print("\nSaving Weights")
        weight_saver.save(filename=args.save_file)
