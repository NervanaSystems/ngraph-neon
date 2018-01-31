# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
import collections
import numpy as np
from ngraph.transformers.base import Computation
from ngraph.transformers.base import ComputationGraphTransformer
from ngraph.op_graph.op_graph import Op
from orderedset import OrderedSet
from ngraph.transformers.passes.pybindwrapperpass \
    import PybindWrapperGenerator
import pyngraph.util as util
from pyngraph import Type, Function
from pyngraph.runtime import Manager


class PybindComputation(Computation):

    def __init__(self, transformer, computation_op, **kwargs):
        super(PybindComputation, self).__init__(transformer, computation_op, **kwargs)
        self.transformer = transformer
        self.computation_op = computation_op
        self.parameter_list = []
        # self.param_primary_tensor_view_list = []
        self.result_primary_tensor_view_list = []
        self.ngraph_op_result_list = []
        self.return_result_list = dict()

        self.initialize_cpp_backend(self.computation_op.returns, *self.computation_op.parameters)

    def __call__(self, *args, **kwargs):
        """
        This builds a primary_tensor_view from the *args and
        results to pass the ([parameter_list], [return_list])
        to ngraph++ call_frame to compute the results.

        :param args:
        :param kwargs:
        :return: [list of computed results]
        """
        param_primary_tensor_view_list = []
        args = self.unpack_args_or_feed_dict(args, kwargs)
        self.get_ngraph_cpp_param_list(param_primary_tensor_view_list, *args)

        for index, result in enumerate(self.result_primary_tensor_view_list):
            result_op = self.ngraph_op_result_list[index]
            element_size = self.get_element_size(result_op)
            shape = list(result_op.axes.lengths)
            # TODO - need to define dtype of numpy array's for results based on result.dtype
            result_arr = np.empty(shape, dtype=np.float32)
            result.write(util.numpy_to_c(result_arr), 0, int(element_size))
            self.return_result_list[result_op] = result_arr

        self.cf.call(param_primary_tensor_view_list, self.result_primary_tensor_view_list)

        # now read the values from the computed result
        for index, result in enumerate(self.result_primary_tensor_view_list):
            result_op = self.ngraph_op_result_list[index]
            element_size = self.get_element_size(result_op)
            result.read(util.numpy_to_c(self.return_result_list[result_op]), 0, int(element_size))

        # determine whether the value to be retruned is a list, dict or an op.
        if isinstance(self.computation_op.returns, Op):
            return self.return_result_list[self.computation_op.returns]
        elif isinstance(self.computation_op.returns, (collections.Sequence, OrderedSet)):
            return tuple(self.return_result_list[op] for op in self.computation_op.returns)
        elif isinstance(self.computation_op.returns, collections.Set):
            return self.return_result_list
        else:
            return None

    def initialize_cpp_backend(self, results, *parameters):
        """
        Passes result's primary_tensor_view to the ngraph++ Function to generate ngraph ++ op graph
        allocates backend and initilizes ngraph++ call frame.

        :param results:
        :param parameters:
        :return:
        """
        # define the result type

        # TODO define element type based on result.dtype instead of defaulting to f32
        self.result_element_type = Type.f32
        result_nodes_list = []
        result_node_to_shape = dict()

        if isinstance(self.computation_op.returns, Op):
            self.ngraph_op_result_list.append(results)
            result_node_to_shape[results] = list(results.axes.lengths)

            result_nodes_list.append(self.transformer.ngraph_cpp_op_prameter[results])
        else:
            for node in results:
                self.ngraph_op_result_list.append(node)
                result_node_to_shape[node] = list(node.axes.lengths)

                result_nodes_list.append(self.transformer.ngraph_cpp_op_prameter[node])

        # use the ngraph_cpp_op dict to built the parameter list for c++ backend
        for place_holders in parameters:
            self.parameter_list.append(
                self.transformer.ngraph_cpp_op_prameter[place_holders.tensor])

        # TODO - what's the role of the string argument? for now just passing 'test'
        self.function = Function(
            result_nodes_list,
            self.parameter_list,
            'test')

        self.manager = Manager.get(self.transformer.ngraph_backend)
        self.external = self.manager.compile(self.function)
        self.backend = self.manager.allocate_backend()
        self.cf = self.backend.make_call_frame(self.external)
        # create the primary_tensor_view for result's using the ngraph++ initilized backend
        if isinstance(self.computation_op.returns, Op):
            self.result_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    self.result_element_type,
                    result_node_to_shape[results]))
        elif isinstance(self.computation_op.returns, (collections.Sequence, OrderedSet)):
            for node in results:
                self.result_primary_tensor_view_list.append(
                    self.backend.make_primary_tensor_view(
                        self.result_element_type, result_node_to_shape[node]))

    def get_ngraph_cpp_param_list(self, param_primary_tensor_view_list, *args):
        """
        Builds a list of ngraph++ primary_tensor_view from ngraph *parameters list

        :param args:
        :return:
        """
        # get the primary tensor_view for all the *parameters passed from the user
        for op in self.computation_op.parameters:
            # TODO define element type based on op.dtype instead of deafulting to flaot32
            param_element_type = Type.f32
            param_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    param_element_type, list(
                        op.axes.lengths)))

        # use c++ backend write method to pass the tensor values
        for index, op in enumerate(self.computation_op.parameters):
            element_size = self.get_element_size(op)
            # TODO - need to define dtype of numpy array's for *params based on op.dtype
            param_primary_tensor_view_list[index].write(util.numpy_to_c(
                np.array(args[index], dtype=np.float32)), 0, int(element_size))

    def get_element_size(self, op):
        """
        computes the size of the op in bytes

        :param op:
        :return: int, op size in bytes
        """
        item_size = op.tensor.dtype.itemsize
        element_size = (np.prod(op.axes.lengths)) * item_size
        return element_size


class PybindTransformer(ComputationGraphTransformer):
    """
    Transformer for executing graphs to call the pybind wrapper of the ngraph c++.

    """
    """
    transformer_name = "pybind_translator"
    """

    def __init__(self, **kwargs):
        """
        if "backend" in kwargs:
            self.ngraph_backend = kwargs.pop("backend")
        else:
            raise AssertionError("No backend info found, please provide the backend info \
            while creating the transformer_factory()")
        """
        super(PybindTransformer, self).__init__(**kwargs)
        self.graph_passes = []
        self.graph_passes += [PybindPregenPass(self)]
        self.graph_passes += [PybindWrapperGenerator(self)]
        self.computation_op_list = []
        self.ngraph_cpp_op_prameter = dict()

    def add_computation(self, computation):
        """
        Initilize ngraph++ backend and call's make_computation to generate PybindComputation Obj

        :param computation:
        :param results:
        :param parameters:
        :return: PybindComputation() object
        """
        self.computation_op_list = OrderedSet()
        if isinstance(computation.returns, collections.Container):
            self.computation_op_list.update(list(computation.returns))
        elif isinstance(computation.returns, Op):
            self.computation_op_list.update(list([computation.returns]))
        self.run_registered_graph_passes(self.computation_op_list)
        return self.make_computation(computation)

    def make_computation(self, computation):
        """
        creates PybindComputation object

        :param computation:
        :return: instance of PybindComputation()
        """
        pybind_comp = PybindComputation(self, computation)
        return pybind_comp

    def initialize(self):
        pass

    def device_buffer_storage(self, bytes, dtype, name):
        pass

    def start_transform_allocate(self):
        pass

    def transform_allocate_ops(self, all_ops):
        pass

    def finish_transform_allocate(self):
        pass

    def transform_ordered_ops(self, ordered_ops, name):
        pass

    def finish_transform(self):
        pass

    def allocate_storage(self):
        pass

    def add_initialization_ops(self, ops):
        pass

    def state_initializations(self, states):
        pass


class PybindCPUTransformer(PybindTransformer):
    """
    Transformer for ngraph c++ with cpu backend.

    """
    transformer_name = "ngcpu"

    def __init__(self, **kwargs):
        self.ngraph_backend = "CPU"
        super(PybindCPUTransformer, self).__init__(**kwargs)


class PybindINTERPRETERTransformer(PybindTransformer):
    """
    Transformer for ngraph c++ with interpreter backend.

    """
    transformer_name = "nginterp"

    def __init__(self, **kwargs):
        self.ngraph_backend = "INTERPRETER"
        super(PybindINTERPRETERTransformer, self).__init__(**kwargs)


class PybindGPUTransformer(PybindTransformer):
    """
    Transformer for ngraph c++ with interpreter backend.

    """
    transformer_name = "nggpu"

    def __init__(self, **kwargs):
        self.ngraph_backend = "GPU"
        super(PybindGPUTransformer, self).__init__(**kwargs)
