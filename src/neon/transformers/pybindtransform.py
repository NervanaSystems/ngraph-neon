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

import collections
import numpy as np
from neon.transformers.base import Computation
from neon.transformers.base import Transformer
from neon.transformers import set_transformer_factory, make_transformer_factory
from neon.op_graph.op_graph import Op, AssignableTensorOp, TensorValueOp, SequentialOp, \
    AssignOp
from neon.op_graph.batchnorm import BatchnormCommonOp, BatchnormBpropCommonOp
from orderedset import OrderedSet
from neon.transformers.passes.pybindwrapperpass \
    import PybindWrapperGenerator, PybindScopePass
from ngraph.impl import util, serialize
from ngraph.impl import Type, Function, NodeVector, Shape
from ngraph.impl.runtime import Manager
from ngraph.impl.op import Parameter


class PybindComputation(Computation):

    def __init__(self, transformer, computation_op, **kwargs):
        super(PybindComputation, self).__init__(transformer, computation_op, **kwargs)
        self.transformer = transformer
        self.computation_op = computation_op

        # Interfacing with Ngraph Function
        # input to Function
        self.parameter_list = []
        self.variable_list = []
        self.randomvariable_list = []
        # result from Function
        self.result_nodes_list = []
        self.update_nodes_list = []

        # Interfacing with Ngraph callframe
        # input to callframe
        self.param_primary_tensor_view_list = []
        self.variable_primary_tensor_view_list = []
        self.randomvariable_primary_tensor_view_list = []
        # output from callframe
        self.result_primary_tensor_view_list = []
        self.update_primary_tensor_view_list = []

        # Interfacing with Neon Ops
        self.neon_variable_list = []
        self.neon_randomvariable_list = []
        self.neon_update_list = []
        self.neon_return_list = []

        # neon side numpy buffer management
        self.neon_return_buffer = dict()
        self.neon_update_buffer = dict()

        # Neon -> Ngraph lookup
        self.ngraph_cpp_ops = dict()
        self.variables_cpp_op = dict()

        # Other variables and structures
        self.op_rank = dict()
        self.rank = 0
        self.scopevisited = set()
        self.scopemark = dict()
        self.seqcount = 0
        self.parcount = 0

        self.function_count = 0
        self.build_opgraph()
        self.build_function()
        self.build_callframe()

    def __call__(self, *args, **kwargs):
        """
        Build primary_tensor_view for parameters ([parameter_list])
        from *args for placeholder and input weight tensors.
        Build primary_tensor_view for return values ([return_list])
        from output computations and updated weights
        Call call_frame with ([parameter_list], [return_list])
        and copy back return values, updated weights.

        :param args:
        :param kwargs:
        :return: [list of computed results]
        """
        args = self.unpack_args_or_feed_dict(args, kwargs)
        # set tensor values for placeholders from args
        # use c++ backend write method to pass the tensor values
        for index, op in enumerate(self.computation_op.parameters):
            if isinstance(args[index], np.ndarray):
                if args[index].dtype == np.float32:
                    input_arg = args[index]
                else:
                    input_arg = args[index].astype(dtype=np.float32)
                    op.tensor.dtype = np.dtype('float32')
            else:
                input_arg = np.array(args[index], dtype=np.float32)
            tensor_size = self.get_tensor_size(op)
            # print("Parameter " + op.tensor.name + " " + str(tensor_size))
            # TODO - need to define dtype of numpy array's for *params based on op.dtype
            self.param_primary_tensor_view_list[index].write(util.numpy_to_c(
                input_arg), 0, tensor_size)
            if not op.tensor.is_placeholder:
                np.copyto(self.transformer.neon_variable_buffer[op.tensor], input_arg)

        # set tensor values for weights from variable buffer
        index = 0
        for op in self.neon_variable_list:
            if op not in self.computation_op.parameters:
                tensor_size = self.get_tensor_size(op)
                # print("In Variable " + op.name + " " + str(tensor_size))
                # TODO - need to define dtype of numpy array's for *params based on op.dtype
                self.variable_primary_tensor_view_list[index].write(util.numpy_to_c(
                    self.transformer.neon_variable_buffer[op]), 0, tensor_size)
                index += 1
        # set tensor values for random variables
        index = 0
        for op in self.neon_randomvariable_list:
            tensor_size = self.get_tensor_size(op)
            size = op.axes.lengths
            distribution = op.distribution
            randparams = op.params
            if distribution == 'uniform':
                randval = np.random.uniform(low=randparams['low'],
                                            high=randparams['high'],
                                            size=size).astype(dtype=np.float32)
            elif distribution == 'normal':
                randval = np.random.normal(loc=randparams['loc'],
                                           scale=randparams['scale'],
                                           size=size).astype(dtype=np.float32)
            else:
                raise ValueError((
                    'unsupported distribution: {}'
                ).format(distribution))
            self.randomvariable_primary_tensor_view_list[index].write(util.numpy_to_c(
                randval), 0, tensor_size)
            index += 1
        """
        print("Var In:")
        for var in self.transformer.neon_variable_buffer:
            if var.name.startswith('GradientDescentMomentum/Sequential/Affine/Linear/W'):
                print("In: " + var.name)
                print(self.transformer.neon_variable_buffer[var])
        """
        self.cf.call(self.result_primary_tensor_view_list + self.update_primary_tensor_view_list,
                     self.param_primary_tensor_view_list + self.variable_primary_tensor_view_list + self.randomvariable_primary_tensor_view_list)

        # now read the values from the computed result
        for index, result in enumerate(self.result_primary_tensor_view_list):
            result_op = self.neon_return_list[index]
            tensor_size = self.get_tensor_size(result_op)
            # print("Result " + result_op.name + " " + str(tensor_size))
            result.read(util.numpy_to_c(self.neon_return_buffer[result_op]),
                        0,
                        tensor_size)
            # print(self.neon_return_buffer[result_op])

        # now read updated weights into weight variables from the computated result
        for index, result in enumerate(self.update_primary_tensor_view_list):
            result_op = self.neon_update_list[index]
            tensor_size = self.get_tensor_size(result_op)
            # print("Out variable " + result_op.name + " " + str(tensor_size))
            result.read(util.numpy_to_c(self.neon_update_buffer[result_op]),
                        0,
                        tensor_size)

        """
        for var in self.neon_update_buffer:
            print("Variable update: " + var.name)
            print(self.neon_update_buffer[var])
        """

        # update weights
        for var in self.neon_update_buffer:
            np.copyto(self.transformer.neon_variable_buffer[var], self.neon_update_buffer[var])

        # determine whether the value to be retruned is a list, dict or an op.
        if isinstance(self.computation_op.returns, Op):
            return self.neon_return_buffer[self.computation_op.returns]
        elif isinstance(self.computation_op.returns, (collections.Sequence, OrderedSet)):
            return tuple(self.neon_return_buffer[op] for op in self.computation_op.returns)
        elif isinstance(self.computation_op.returns, collections.Set):
            return self.neon_return_buffer
        else:
            return None

    def search_cpp_op(self, op):
        if isinstance(op, SequentialOp):
            op = op.ops[-1]
        if isinstance(op, TensorValueOp):
            tensor_op = op.tensor
            if not tensor_op.is_constant:
                if tensor_op in self.variables_cpp_op:
                    if self.scopemark[op] == self.variables_cpp_op[tensor_op][0]:
                        if self.op_rank[op] > self.op_rank[self.variables_cpp_op[tensor_op][1]]:
                            """
                            print("Forwarding " + tensor_op.name + " to " + op.name +
                                  " forward value " + self.variables_cpp_op[tensor_op][1].name)
                            """
                            return self.ngraph_cpp_ops[self.variables_cpp_op[tensor_op][1].tensor]
        else:
            tensor_op = op.tensor
        if tensor_op in self.ngraph_cpp_ops:
            return self.ngraph_cpp_ops[tensor_op]
        return None

    def lookup_cpp_op(self, op):
        ngraph_op = self.search_cpp_op(op)
        if ngraph_op is None:
            raise RuntimeError("Ngraph Op missing for Neon Op " + op.name)
        else:
            return ngraph_op

    def has_cpp_op(self, op):
        ngraph_op = self.search_cpp_op(op)
        if ngraph_op is None:
            return False
        else:
            return True

    def register_cpp_op(self, op, cpp_op, set_name=True):
        if isinstance(op, SequentialOp):
            if hasattr(op, 'axes'):
                neon_shape = list(op.axes.lengths)
                ngraph_shape = [cpp_op.shape[i] for i in range(len(cpp_op.shape))]
                if neon_shape != ngraph_shape:
                    raise RuntimeError("Shape mismatch", op.name, neon_shape, ngraph_shape)
            self.ngraph_cpp_ops[op] = cpp_op
            return

        if isinstance(op, AssignableTensorOp):
            """
            print("register_cpp_op: " + op.name + \
                  ", is_constant: " + str(op.is_constant) + \
                  ", is_trainable: " + str(op.is_trainable) + \
                  ", is_placeholder: " + str(op.is_placeholder))
            """
            tensor_op = op
        else:
            tensor_op = op.tensor

        if tensor_op in self.ngraph_cpp_ops:
            raise RuntimeError("Cannot register neon op twice: " + tensor_op.name)

        if set_name:
            try:
                cpp_op.name = tensor_op.name.replace('/', '_')
            except RuntimeError:
                pass
        if not isinstance(op, (BatchnormCommonOp, BatchnormBpropCommonOp)):
            if hasattr(op, 'axes'):
                neon_shape = list(op.axes.lengths)
                ngraph_shape = [cpp_op.shape[i] for i in range(len(cpp_op.shape))]
                if neon_shape != ngraph_shape:
                    raise RuntimeError("Shape mismatch", op.name, neon_shape, ngraph_shape)
        self.ngraph_cpp_ops[tensor_op] = cpp_op

    def set_op_rank(self, op):
        if isinstance(op, TensorValueOp):
            self.op_rank[op] = self.rank
        else:
            self.op_rank[op.tensor] = self.rank
        self.rank += 1

    def get_tensor_size(self, op):
        """
        computes the size of the tensor produced by op in bytes

        :param op:
        :return: int, op size in bytes
        """
        if isinstance(op, AssignOp):
            op = op.args[1]
        item_size = op.tensor.dtype.itemsize
        tensor_size = int((np.prod(op.axes.lengths)) * item_size)
        return tensor_size

    def build_opgraph(self):
        """
        Build Ngraph opgraph from Neon's computation graph.
        """
        computation = self.computation_op
        self.transformer.graph_passes = []
        self.transformer.graph_passes += [PybindWrapperGenerator(self.transformer, self)]
        self.custom_passes = []
        self.custom_passes += [PybindScopePass(self)]
        computation_op_list = OrderedSet()
        if isinstance(computation.returns, collections.Container):
            computation_op_list.update(list(computation.returns))
        elif isinstance(computation.returns, Op):
            computation_op_list.update(list([computation.returns]))
        for custom_pass in self.custom_passes:
            custom_pass(computation_op_list)
        self.transformer.run_registered_graph_passes(computation_op_list)

    def build_function(self):
        """
        Build Ngraph Function from opgraph.
        """
        # define the result type

        # TODO define element type based on dtype instead of defaulting to f32
        self.element_type = Type.f32

        if isinstance(self.computation_op.returns, Op):
            self.neon_return_list.append(self.computation_op.returns)
        else:
            self.neon_return_list = self.computation_op.returns

        for node in self.neon_return_list:
            # print("Result: " + node.name)
            if isinstance(node.tensor, AssignOp):
                node = node.args[1]
            ngraph_op = self.lookup_cpp_op(node)
            # print("Return " + str(ngraph_op))
            self.result_nodes_list.append(ngraph_op)

        # Add additional results (updated variable)
        for variable in self.variables_cpp_op:
            # print("Update " + variable.name + " " + self.variables_cpp_op[variable][1].name)
            self.neon_update_list.append(variable)
            ngraph_op = self.lookup_cpp_op(self.variables_cpp_op[variable][1])
            # print("Outvar " + str(ngraph_op))
            """
            rhs = self.variables_cpp_op[variable][1].tensor
            for op in Op.ordered_ops([rhs]):
                print("    " + op.name)
                for arg in op.args:
                    print("        " + arg.name)
            """
            self.update_nodes_list.append(ngraph_op)
            shape = list(variable.axes.lengths)
            self.neon_update_buffer[variable] = np.zeros(shape, dtype=np.float32)

        # use the ngraph_cpp_op dict to built the parameter list for c++ backend
        for place_holders in self.computation_op.parameters:
            if place_holders.tensor in self.ngraph_cpp_ops:
                self.parameter_list.append(self.ngraph_cpp_ops[place_holders.tensor])
            else:  # sometimes parameters can be unused/dead values in computation.
                tensor = place_holders.tensor
                op_element_type = Parameter(Type.f32, Shape(list(tensor.axes.lengths)))
                self.register_cpp_op(tensor, op_element_type)
                if not tensor.is_placeholder:
                    self.neon_variable_list.append(tensor)
                self.parameter_list.append(op_element_type)

        # Add additional parameters (variables)
        for variable in self.neon_variable_list:
            if variable not in self.computation_op.parameters:
                self.variable_list.append(self.ngraph_cpp_ops[variable.tensor])
            # Allocate variable buffer - shared by computations
            # TODO - need to define dtype of numpy array's for variables based on dtype
            if variable not in self.transformer.neon_variable_buffer:
                shape = list(variable.axes.lengths)
                var_buffer = np.zeros(shape, dtype=np.float32)
                self.transformer.neon_variable_buffer[variable] = var_buffer
                if variable.initial_value is not None:
                    np.copyto(var_buffer, variable.initial_value)

        # Add additional parameters (random numbers)
        for randvariable in self.neon_randomvariable_list:
            self.randomvariable_list.append(self.ngraph_cpp_ops[randvariable.tensor])

        # TODO - what's the role of the string argument? for now just passing 'test'
        funcname = self.transformer.get_function_name()
        self.function = Function(NodeVector(
            self.result_nodes_list + self.update_nodes_list),
            self.parameter_list + self.variable_list + self.randomvariable_list,
            funcname)
        
        # Serialize function
        """
        jsonstr = serialize(self.function)
        filename = funcname + '.json'
        with open(filename, 'w') as f:
            f.write(jsonstr)
        """

    def build_callframe(self):
        """
        Initialize Ngraph backend. Build and initialize Ngraph callframe from Function.
        """
        self.manager = Manager.get(self.transformer.ngraph_backend)
        self.external = self.manager.compile(self.function)
        self.backend = self.manager.allocate_backend()
        self.cf = self.backend.make_call_frame(self.external)

        # create the primary_tensor_view for result's using the ngraph++ initilized backend
        for node in self.neon_return_list:
            org_node = node
            if isinstance(node.tensor, AssignOp):
                node = node.args[1]
            shape = list(node.tensor.axes.lengths)
            self.result_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    self.element_type, Shape(shape)))
            # Allocate return buffer
            # TODO - need to define dtype of numpy array's for results based on result.dtype
            result_arr = np.zeros(shape, dtype=np.float32)
            self.neon_return_buffer[org_node] = result_arr

        # prepare tensor_views for placeholders
        for node in self.computation_op.parameters:
            shape = list(node.axes.lengths)
            self.param_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    self.element_type, Shape(shape)))

        # prepare tensor_views for input variables
        for node in self.neon_variable_list:
            if node not in self.computation_op.parameters:
                shape = list(node.axes.lengths)
                self.variable_primary_tensor_view_list.append(
                    self.backend.make_primary_tensor_view(
                        self.element_type, Shape(shape)))

        # prepare tensor_views for input variables
        for node in self.neon_randomvariable_list:
            shape = list(node.axes.lengths)
            self.randomvariable_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    self.element_type, Shape(shape)))

        # prepare tensor_views for weights
        for node in self.neon_update_list:
            shape = list(node.axes.lengths)
            self.update_primary_tensor_view_list.append(
                self.backend.make_primary_tensor_view(
                    self.element_type, Shape(shape)))


class FunctionTransformer(Transformer):

    def __init__(self, **kwargs):
        super(FunctionTransformer, self).__init__(**kwargs)
        self.computations = OrderedSet()

    def run_registered_graph_passes(self, ops, **kwargs):
        for graph_pass in self.graph_passes:
            graph_pass.wrapped_do_pass(ops=ops, **kwargs)
        return ops

    def host_to_device(self, computation, parameters, args):
        pass

    def device_to_host(self, computation, op, tensor=None):
        pass

    def get_tensor_view_value(self, op, host_tensor=None):
        pass

    @property
    def use_exop(self):
        """

        Returns: True if this transformer uses the execution graph.

        """
        return False

    def initialize_allocations(self):
        pass

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


class PybindTransformer(FunctionTransformer):
    """
    Transformer for executing graphs to call the pybind wrapper of the ngraph c++.

    """
    """
    transformer_name = "pybind_translator"
    """
    function_count = 1

    def __init__(self, **kwargs):
        """
        if "backend" in kwargs:
            self.ngraph_backend = kwargs.pop("backend")
        else:
            raise AssertionError("No backend info found, please provide the backend info \
            while creating the transformer_factory()")
        """
        super(PybindTransformer, self).__init__(**kwargs)
        self.neon_variable_buffer = dict()

    def make_computation(self, computation):
        """
        creates PybindComputation object

        :param computation:
        :return: instance of PybindComputation()
        """
        pybind_comp = PybindComputation(self, computation)
        return pybind_comp

    def get_function_name(self):
        name = 'function' + str(PybindTransformer.function_count)
        PybindTransformer.function_count += 1
        return name


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
    Transformer for ngraph c++ with gpu backend.

    """
    transformer_name = "nggpu"

    def __init__(self, **kwargs):
        self.ngraph_backend = "GPU"
        super(PybindGPUTransformer, self).__init__(**kwargs)


class PybindNNPTransformer(PybindTransformer):
    """
    Transformer for ngraph c++ with nnp backend.

    """
    transformer_name = "ngnnp"

    def __init__(self, **kwargs):
        self.ngraph_backend = "NNP"
        super(PybindNNPTransformer, self).__init__(**kwargs)


set_transformer_factory(
    make_transformer_factory(PybindCPUTransformer.transformer_name))
