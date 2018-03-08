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
from __future__ import division, print_function, absolute_import

import collections
from operator import itemgetter
import neon as ng
from neon.frontend.saverfile import SaverFile


def get_root_ops(computation):
    """
    Get list of root Ops from a forest of computation graph

    Arguments:
        computation : All outputs of computation (An Op, list of Ops or dictionary of Ops)
    Returns:
        List of root Ops
    """
    # Handle arg to neon.make_bound_computation()
    if isinstance(computation, dict):
        computation_keys = tuple(sorted(computation.keys()))
        outputs = itemgetter(*computation_keys)(computation)
        outputs = [outputs] if len(computation_keys) == 1 else list(outputs)
        values = type(outputs)(ng.as_op(output) for output in outputs)
    # Handle arg to transformer.add_computation()
    elif isinstance(computation, ng.ComputationOp):
        values = computation.values
    # Handle arg to transformer.compuation()
    elif isinstance(computation, collections.Iterable):
        values = list(computation)
    else:
        raise ValueError()
    return values


class Saver(object):
    def __init__(self):
        """
        A class that defines a set of methods to enable weight saving and restoring

        Methods:
            setup_save: prepare save function for saving all weight variables in
                        computation
            save: saves weight values to named file
            setup_restore: prepare restore function for loading weight from file to
                           weight variables in computation
            restore: load weight values to computation
        Examples:
            ... create some_op_graph ...
            comp = ng.computation(some_op_graph, "all")

            " create saver object
            weight_saver = Saver()
            with closing(ngt.make_transformer()) as transformer:
                func = transformer.add_computation(comp)
                " setup save function
                weight_saver.setup_save(transformer=transformer, computation=comp)
                ... some usage of func ...
                " call save
                weight_saver.save(filename="some_name")
            ...
            with closing(ngt.make_transformer()) as another_transformer:
                another_func = restore_transformer.add_computation(comp)
                " setup restore
                weight_saver.setup_restore(transformer=another_transformer,
                                           computation=comp,
                                           filename="some_name")
                " call restore
                weight_saver.restore()
                ... now use another_func with the restored weights ...
        """
        self.getter_op_names = None
        self.getter = None
        self.getter_type = 'new'
        self.setter = None
        self.setter_type = 'new'

    def setup_save(self, transformer, computation):
        """
        prepare save function for saving all weight variables in computation

        Arguments:
            transformer : transformer where the weights are stored
            computation : All outputs of computation (An Op, list of Ops or dictionary of Ops)
        """
        # collect and return a set of all Variables
        def find_ops(values):
            """
            Find and return all weights.
            """
            nodes = dict()
            frontier = set(values)
            visited = set()

            def find_op(op_to_add):
                """
                find weight (trainable AssignableTensorOp)
                """
                tensor = op_to_add.tensor
                if tensor.is_persistent and not (tensor.is_constant or tensor.is_placeholder):
                    try:
                        prev_op = nodes[tensor.name]
                    except KeyError:
                        prev_op = tensor
                        nodes[tensor.name] = tensor
                    assert prev_op == tensor

            while len(frontier) > 0:
                op_to_visit = frontier.pop()
                find_op(op_to_visit)
                visited.add(op_to_visit)
                for arg in op_to_visit.args:
                    if arg not in visited:
                        frontier.add(arg)
                for arg in op_to_visit.all_deps:
                    if arg not in visited:
                        frontier.add(arg)
            return nodes
        # Traverse computation graph and extract persistent tensors and unique op instance name
        save_variables = find_ops(get_root_ops(computation))
        self.getter_op_names, ops = zip(*save_variables.items())
        if transformer.transformer_name not in ("ngcpu", "nginterp", "nggpu"):
            self.getter = transformer.computation(ops)
            self.getter_type = 'old'
        else:
            self.getter = transformer.neon_variable_buffer
            self.getter_type = 'new'

    def save(self, filename, compress=False, transformer=None, computation=None):
        """
        Save weight values to named file

        Arguments:
            filename: name of file to be used for saving weights
            compress: specify whether to compress the weights
            transformer : transformer where the weights are stored
                          required only if setup_save is not called
            computation : All outputs of computation (An Op, list of Ops or dictionary of Ops)
                          required only if setup_save is not called
        """
        if self.getter is None:
            self.setup_save(transformer=transformer,
                            computation=computation)
        tensors = dict()
        if self.getter_type == 'old':
            tensors = {name: tensor.copy() for name, tensor in zip(self.getter_op_names,
                                                                   self.getter())}
        else:
            for op, tensor in self.getter.items():
                tensors[op.name] = tensor
        # write dictionary to file
        savefile = SaverFile(filename)
        savefile.write_values(tensors, compress)

    def setup_restore(self, transformer, computation, filename):
        """
        prepare restore function for loading weight from file to
        weight variables in computation

        Arguments:
            transformer : transformer where the weights will be restored
            computation : All outputs of computation (An Op, list of Ops or dictionary of Ops)
            filename: name of file with saved weights
        """
        def match_ops(tensors, values):
            """
            Match weights with tensor values loaded from file
            """
            nodes = dict()
            frontier = set(values)
            visited = set()

            def match_op(op_to_add):
                """
                Match a weight with loaded tensor value
                """
                tensor = op_to_add.tensor
                if tensor.is_persistent and not (tensor.is_constant or tensor.is_placeholder):
                    try:
                        nodes[tensor] = tensors[tensor.name]
                    except KeyError:
                        print("Warning: Missing weight in save file: " + tensor.name)

            while len(frontier) > 0:
                op_to_visit = frontier.pop()
                match_op(op_to_visit)
                visited.add(op_to_visit)
                for arg in op_to_visit.args:
                    if arg not in visited:
                        frontier.add(arg)
                for arg in op_to_visit.all_deps:
                    if arg not in visited:
                        frontier.add(arg)
            return nodes
        # load weight from file to tensors
        savefile = SaverFile(filename)
        tensors = savefile.read_values()
        nodes = match_ops(tensors, get_root_ops(computation))
        if transformer.transformer_name not in ("ngcpu", "nginterp", "nggpu"):
            restore_ops = []
            for op_to_save, op_value in nodes.items():
                restore_ops.append(ng.AssignOp(op_to_save, op_value))
            self.setter = transformer.computation(restore_ops)
            self.setter_type = 'old'
        else:
            self.setter = (nodes, transformer.neon_variable_buffer)
            self.setter_type = 'new'

    def restore(self, transformer=None, computation=None, filename=None):
        """
        load weight values to computation
        Arguments:
            transformer : transformer where the weights will be restored
                          required only if setup_restore is not called
            computation : All outputs of computation (An Op, list of Ops or dictionary of Ops)
                          required only if setup_restore is not called
            filename: name of file with saved weights
                      required only if setup_restore is not called
        """
        if self.setter is None:
            self.setup_restore(transformer=transformer,
                               computation=computation,
                               filename=filename)
        if self.setter_type == 'old':
            self.setter()
        else:
            nodes = self.setter[0]
            for op in nodes:
                self.setter[1][op] = nodes[op]
