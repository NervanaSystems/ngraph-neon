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
import abc
import itertools

from future.utils import with_metaclass

from neon.op_graph.axes import make_axis
from neon.transformers.passes.opdelegate import DelegateOpAccessor
from neon.util.generics import generic_method


class GraphPass(with_metaclass(abc.ABCMeta, DelegateOpAccessor)):
    def wrapped_do_pass(self, **kwargs):
        self.begin_pass(**kwargs)
        self.do_pass(**kwargs)
        self.end_pass(**kwargs)

    @abc.abstractmethod
    def do_pass(self, **kwargs):
        pass


class ProcessOpGraphPass(GraphPass):
    def do_pass(self, **kwargs):
        self.run_pass(self.process_op, **kwargs)

    @abc.abstractmethod
    def process_op(self, op):
        """
        Called from run_pass to perform processing on an op-graph Op op.

        Args:
            op: The op-graph Op to be

        Returns:

        """


class GraphBuildingPass(ProcessOpGraphPass):

    def process_op(self, op):
        self.visit(op, *self.op_args(op))


class PeepholeGraphPass(GraphBuildingPass):
    """
    Base class for passes that do not add to the graph.

    TODO: currently it has same exact implementation as GraphBuildingPass,
    consider removing it in future.
    """
    pass


