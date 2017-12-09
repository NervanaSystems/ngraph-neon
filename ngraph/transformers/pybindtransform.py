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
from ngraph.transformers.base import make_transformer_factory, set_transformer_factory
from ngraph.op_graph.op_graph import Op, computation
from orderedset import OrderedSet
from ngraph.transformers.passes.pybindwrapperpass import PybindWrapperGenerator
import nwrapper.ngraph.Util as Util
import nwrapper.ngraph.types.TraitedType as TraitedType
import nwrapper.ngraph.types.TensorViewType as TensorViewType
import nwrapper.ngraph.Function as Function
import nwrapper.ngraph.runtime.Manager as Manager


class PybindComputation(Computation):

    def __init__(self, transformer, computation_op, results, *parameters, **kwargs):
        super(PybindComputation, self).__init__(transformer, computation_op, **kwargs)
        self.transformer = transformer
        self.computation_op = computation_op
        self.parameter_list = []
        self.param_primary_tensor_view_list = []
        self.result_primary_tensor_view_list = []
        self.ngraph_op_result_list = []
        self.return_result_list = []

        self.ngraph_op_result_list.append(results)
        self.initialize_cpp_backend(results, *parameters)

    def __call__(self, *args, **kwargs):
        """
        This builds a primary_tensor_view from the *args and
        results to pass the ([parameter_list], [return_list])
        to ngraph++ call_frame to compute the results.

        :param args:
        :param kwargs:
        :return: [list of computed results]
        """
        args = self.unpack_args_or_feed_dict(args, kwargs)
        self.get_ngraph_cpp_param_list(*args)

        for index, result in enumerate(self.result_primary_tensor_view_list):
            element_size = self.get_element_size(self.ngraph_op_result_list[index])
            shape = list(self.ngraph_op_result_list[index].axes.lengths)
            # TODO - need to define dtype of numpy array's for results based on result.dtype
            result_arr = np.empty(shape, dtype=np.float32)
            result.write(Util.numpy_to_c(result_arr), 0, int(element_size))
            self.return_result_list.append(result_arr)

        self.cf.call(self.param_primary_tensor_view_list, self.result_primary_tensor_view_list)

        # now read the values from the computed result
        for index, result in enumerate(self.result_primary_tensor_view_list):
            element_size = self.get_element_size(self.ngraph_op_result_list[index])
            result.read(Util.numpy_to_c(self.return_result_list[index]), 0, int(element_size))

        return self.return_result_list

    def initialize_cpp_backend(self, results, *parameters):
        """
        Passes result's primary_tensor_view to the ngraph++ Function to generate ngraph ++ op graph
        allocates backend and initilizes ngraph++ call frame.

        :param results:
        :param parameters:
        :return:
        """
        # define the result type
        # TODO define element type based on result.dtype instead of deafulting to flaot32
        self.result_element_type = TraitedType.TraitedTypeF.element_type()
        if not(isinstance(results, list)):
            self.result_shape = list(results.axes.lengths)
        value_type = TensorViewType.TensorViewType(self.result_element_type, self.result_shape)

        # use the ngraph_cpp_op dict to built the parameter list for c++ backend
        for place_holders in parameters:
            self.parameter_list.append(self.transformer.ngraph_cpp_op_prameter[place_holders])

        # TODO - what's the role of the string argument? for now just passing 'test'
        self.function = Function.Function(
            self.transformer.ngraph_cpp_op_prameter[results],
            value_type,
            self.parameter_list,
            'test')

        self.manager = Manager.Manager.get(self.transformer.ngraph_backend)
        self.external = self.manager.compile(self.function)
        self.backend = self.manager.allocate_backend()
        self.cf = self.backend.make_call_frame(self.external)
        # create the primary_tensor_view for result's using the ngraph++ initilized backend
        self.result_primary_tensor_view_list.append(self.backend.make_primary_tensor_view
                                                    (self.result_element_type, self.result_shape))

    def get_ngraph_cpp_param_list(self, *args):
        """
        Builds a list of ngraph++ primary_tensor_view from ngraph *parameters list

        :param args:
        :return:
        """
        # get the primary tensor_view for all the *parameters passed from the user
        for op in self.computation_op.parameters:
            # TODO define element type based on op.dtype instead of deafulting to flaot32
            param_element_type = TraitedType.TraitedTypeF.element_type()
            self.param_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    param_element_type, list(
                        op.axes.lengths)))

        # use c++ backend write method to pass the tensor values
        for index, op in enumerate(self.computation_op.parameters):
            element_size = self.get_element_size(op)
            # TODO - need to define dtype of numpy array's for *params based on op.dtype
            self.param_primary_tensor_view_list[index].write(Util.numpy_to_c(
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
    transformer_name = "pybind_translator"

    def __init__(self, **kwargs):
        if "backend" in kwargs:
            self.ngraph_backend = kwargs.pop("backend")
        else:
            raise AssertionError("No backend info found, please provide the backend info \
            while creating the transformer_factory()")
        super(PybindTransformer, self).__init__(**kwargs)
        self.graph_passes = []
        self.graph_passes += [PybindWrapperGenerator(self)]
        self.computation_op_list = []
        self.ngraph_cpp_op_prameter = dict()

    def computation(self, results, *parameters):
        return self.add_computation(computation(results, *parameters), results, *parameters)

    def add_computation(self, computation, results, *parameters):
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
        return self.make_computation(computation, results, *parameters)

    def make_computation(self, computation, results, *parameters):
        """
        creates PybindComputation object

        :param computation:
        :return: instance of PybindComputation()
        """
        pybind_comp = PybindComputation(self, computation, results, *parameters)
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


set_transformer_factory(
    make_transformer_factory(PybindTransformer.transformer_name))
