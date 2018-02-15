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
from __future__ import division
from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, Add, Multiply, BroadcastOp, TensorValueOp, \
    DotOp, LogOp, ExpOp, Sum, Greater, GreaterEqual, Maximum, ReductionOp, AssignableTensorOp, \
    OneHotOp, Divide, Subtract, NegativeOp, ReciprocalOp, TensorSizeOp, MapRolesOp, Minimum, \
    Less, Max, NotEqual, SequentialOp, AssignOp, ParallelOp, ExpandDims, TensorSliceOp, \
    Equal, SqrtOp, SquareOp, Flatten, Unflatten, ContiguousOp, Prod, ReorderAxes
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
import numpy as np

from pyngraph import Type
from pyngraph import Function as Function
from pyngraph.op import Parameter
from pyngraph.op import AvgPool as PyngAvgPool
from pyngraph.op import AvgPoolBackprop as PyngAvgPoolBackprop
from pyngraph.op import Broadcast as PyngBroadcast
from pyngraph.op import Constant
from pyngraph.op import Convert as PyngConvert
from pyngraph.op import Convolution as PyngConvolution
from pyngraph.op import ConvolutionBackpropData as PyngConvolutionBackpropData
from pyngraph.op import ConvolutionBackpropFilters as PyngConvolutionBackpropFilters
from pyngraph.op import Dot as PyngDot
from pyngraph.op import Equal as PyngEqual
from pyngraph.op import Exp as PyngExp
from pyngraph.op import Greater as PyngGreater
from pyngraph.op import GreaterEq as PyngGreaterEq
from pyngraph.op import Less as PyngLess
from pyngraph.op import Log as PyngLog
from pyngraph.op import Maximum as PyngMaximum
from pyngraph.op import MaxPool as PyngMaxPool
from pyngraph.op import MaxPoolBackprop as PyngMaxPoolBackprop
from pyngraph.op import Minimum as PyngMinimum
from pyngraph.op import Multiply as PyngMultiply
from pyngraph.op import Negative as PyngNegative
from pyngraph.op import NotEqual as PyngNotEqual
from pyngraph.op import OneHot as PyngOneHot
from pyngraph.op import Reshape as PyngReshape
from pyngraph.op import Reduce as PyngReduce
from pyngraph.op import Slice as PyngSlice
from pyngraph.op import Sqrt as PyngSqrt
from pyngraph.op import Sum as PyngSum


class PybindScopePass:
    """
    Graph pass mark Variable version scope
    Track AssignOp, SequentionOp and ParallelOp

    Arguments
        transformer (obj:`Transformer`): The associated transformer.
    """

    def __init__(self, computation, **kwargs):
        self.computation = computation

    # scope rules:
    # - Do a pre-order traversal of op_graph
    #     - Default scope is "root"
    #     - Other scopes are formed by appending enclosing ParallelOp and SequentialOp
    #       to default scope like a posix path
    #     - Tag all ops with the enclosing scope name
    def recordscope(self, results):

        def extend_scope(scope, leaf):
            return scope + '/' + leaf

        def new_seq_scope(scope):
            new_scope = extend_scope(scope, 'seq' + str(self.computation.seqcount))
            self.computation.seqcount += 1
            return new_scope

        def new_par_scope(scope):
            new_scope = extend_scope(scope, 'par' + str(self.computation.parcount))
            self.computation.parcount += 1
            return new_scope

        def visit_pre_order(scope, op):
            if isinstance(op, TensorValueOp):
                if op not in self.computation.scopevisited:
                    self.computation.scopevisited.add(op)
                    self.computation.scopemark[op] = scope
                    return
                else:
                    return
            if isinstance(op, SequentialOp):
                if op not in self.computation.scopevisited:
                    childscope = new_seq_scope(scope)
                    children = op.ops
                    self.computation.scopevisited.add(op)
                    self.computation.scopemark[op] = scope
                    for child in children:
                        visit_pre_order(childscope, child)
                    return
                else:
                    return
            elif isinstance(op, ParallelOp):
                if op not in self.computation.scopevisited:
                    childscope = new_par_scope(scope)
                    children = op.control_deps
                    self.computation.scopevisited.add(op)
                    self.computation.scopemark[op] = scope
                    for child in children:
                        visit_pre_order(childscope, child)
                    return
                else:
                    return

            tensor_op = op.tensor
            if tensor_op not in self.computation.scopevisited:
                childscope = scope
                children = tensor_op.args
                self.computation.scopevisited.add(tensor_op)
                self.computation.scopemark[tensor_op] = scope
                for child in children:
                    visit_pre_order(childscope, child)

        for op in results:
            visit_pre_order('root', op)

        # for key, val in self.computation.scopemark.items():
        #    print(key.name, val)

    def __call__(self, results):
        self.recordscope(results)


class PybindWrapperGenerator(PeepholeGraphPass):
    """
    Graph pass to generate the PybindWrapper's by visiting all the Op's
    needed to compute the results.

    Arguments
        transformer (obj:`Transformer`): The associated transformer.
    """

    def __init__(self, transformer, computation, **kwargs):
        super(PybindWrapperGenerator, self).__init__(**kwargs)
        self.transformer = transformer
        self.computation = computation

    def np_reduction_axis(self, op):
        """
        Returns numpy reduction axis of an op

        Args:
            op: instance of ReductionOp

        Returns:
            tuple of numpy reduction axis
        """
        if not isinstance(op, ReductionOp):
            raise ValueError("Op %s must be an instance of ReductionOp" % op)
        input_axes = op.args[0].axes
        reduction_axes = op.reduction_axes
        try:
            np_axis = tuple([input_axes.index(axis) for axis in reduction_axes])
        except ValueError:
            np_axis = tuple([0, ])
        return np_axis[0] if len(np_axis) == 1 else np_axis

    def get_reduction_axis(self, op):
        """
        Returns int value which is proportional to the same axes shared by the input tensors
        :param op:
        :return: int value
        """
        count_common_axis = 0
        reduction_axes = []
        input1_axes = op.args[0].axes.names
        input2_axes = op.args[1].axes.names
        for axis in input1_axes:
            if axis in input2_axes:
                count_common_axis += 1
                reduction_axes.append(axis)
        return (count_common_axis, tuple(reduction_axes))

    def get_shape_from_axes_order(self, axes_order, input_shape):
        """
        returns the shape of the input for transpose based on the given axes_order
        :param axes_order:
        :param input_shape:
        :return:
        """
        # determine the axis order for the reshape
        reorder_shape = []
        for index in axes_order:
            reorder_shape.append(input_shape[index])
        return reorder_shape

    def get_axes_order_from_axes_name(self, input_axes, reshape_axes):
        reshape_axis_order = []
        for pos, val in enumerate(reshape_axes):
            reshape_axis_order.append(input_axes.index(val))

        return reshape_axis_order

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        self.computation.set_op_rank(op)
        raise RuntimeError("Unsupported op " + str(type(op)))

    @visit.on_type(Add)
    def visit(self, op, x, y):
        self.computation.set_op_rank(op)
        ngraph_cpp_add_op = self.computation.lookup_cpp_op(x) \
            + self.computation.lookup_cpp_op(y)

        self.computation.register_cpp_op(op, ngraph_cpp_add_op)

    @visit.on_type(Divide)
    def visit(self, op, x, y):
        self.computation.set_op_rank(op)
        ngraph_cpp_div_op = self.computation.lookup_cpp_op(x) \
            / self.computation.lookup_cpp_op(y)

        self.computation.register_cpp_op(op, ngraph_cpp_div_op)

    @visit.on_type(Multiply)
    def visit(self, op, x, y):
        self.computation.set_op_rank(op)
        ngraph_cpp_mul_op = self.computation.lookup_cpp_op(x) \
            * self.computation.lookup_cpp_op(y)

        self.computation.register_cpp_op(op, ngraph_cpp_mul_op)

    @visit.on_type(Subtract)
    def visit(self, op, x, y):
        self.computation.set_op_rank(op)
        ngraph_cpp_sub_op = self.computation.lookup_cpp_op(x) \
            - self.computation.lookup_cpp_op(y)

        self.computation.register_cpp_op(op, ngraph_cpp_sub_op)

    @visit.on_type(BroadcastOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        axis_set = set()
        element_type = Type.f32
        # check if the op.args already have Paramterized view type.
        if self.computation.has_cpp_op(op.args[0]):
            op_element_type = self.computation.lookup_cpp_op(op.args[0])
        else:
            op_element_type = Parameter(
                element_type, list(op.args[0].axes.lengths))
        # build axis_set
        broadcast_axes = op.axes.lengths
        broadcast_args_axes = op.args[0].axes.lengths

        for pos, axis in enumerate(broadcast_axes):
            if axis not in broadcast_args_axes:
                axis_set.add(pos)

        self.computation.register_cpp_op(
            op, PyngBroadcast(op_element_type, list(op.axes.lengths), axis_set))

    def flatten(self, container):
        if isinstance(container, (list, tuple)):
            for i in container:
                if isinstance(i, (list, tuple)):
                    for j in self.flatten(i):
                        yield j
                else:
                    yield i
        else:
            yield container

    @visit.on_type(TensorValueOp)
    def visit(self, op):
        self.computation.set_op_rank(op)
        tensor = op.tensor
        if not self.computation.has_cpp_op(op):
            if tensor.is_constant:
                # FIXME: make tensors based on data type
                constant_op = Constant(Type.f32,
                                       list(tensor.axes.lengths),
                                       list(self.flatten(tensor.const.tolist())))
                self.computation.register_cpp_op(tensor, constant_op)
            else:
                op_element_type = Parameter(Type.f32, list(tensor.axes.lengths))
                self.computation.register_cpp_op(tensor, op_element_type)
                if not tensor.is_placeholder:
                    self.computation.neon_variable_list.append(tensor)

    @visit.on_type(AssignableTensorOp)
    def visit(self, op):
        # Can be visited in the most trivial computation we only a variable is created
        if not self.computation.has_cpp_op(op):
            if op.is_constant:
                # FIXME: make tensors based on data type
                constant_op = Constant(Type.f32,
                                       list(op.axes.lengths),
                                       list(self.flatten(op.const.tolist())))

                self.computation.register_cpp_op(op, constant_op)
            else:
                op_element_type = Parameter(Type.f32, list(op.axes.lengths))
                self.computation.register_cpp_op(op, op_element_type)
                if not op.is_placeholder:
                    self.computation.neon_variable_list.append(op)

    @visit.on_type(DotOp)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        # determine the reduction_axes count
        reduction_axes_count, reduction_axes = self.get_reduction_axis(op)

        reshape_input_needed = False
        reshape_output_needed = False

        # check if the input1/input2 needs to be Transposed and if yes, Transpose
        if (len(input1.axes.names) != 0 and len(input2.axes.names) != 0) \
                and (input1.axes.names[-1] != input2.axes.names[0]):
            reshape_input_needed = True
            input1_axes = list((op.x_out_axes + op.reduction_axes).names)
            input2_axes = list((op.reduction_axes + op.y_out_axes).names)
            input1_axes_order = self.get_axes_order_from_axes_name(
                input1.axes.names, input1_axes)
            input1_axes_shape = self.get_shape_from_axes_order(
                                    input1_axes_order,
                                    input1.axes.lengths)
            input2_axes_order = self.get_axes_order_from_axes_name(
                input2.axes.names, input2_axes)
            input2_axes_shape = self.get_shape_from_axes_order(
                                    input2_axes_order,
                                    input2.axes.lengths)
        else:
            input1_axes_shape = list(input1.axes.lengths)
            input1_axes_order = list(range(0, len(input1.axes)))
            input2_axes_shape = list(input2.axes.lengths)
            input2_axes_order = list(range(0, len(input2.axes)))

        # flatten reduction_axes
        if reduction_axes_count > 1:
            reshape_input_needed = True
            input1_axes_shape = input1_axes_shape[:-reduction_axes_count] + \
                [np.prod(input1_axes_shape[-reduction_axes_count:])]
            input2_axes_shape = [np.prod(input2_axes_shape[:reduction_axes_count])] + \
                input2_axes_shape[reduction_axes_count:]
            reduction_axes_count = 1

        # check if other axes need to be flatten to force 2D dot
        if reduction_axes_count == 1:
            if len(op.x_out_axes) > 1:
                reshape_input_needed = True
                reshape_output_needed = True
                input1_axes_shape = [np.prod(input1_axes_shape[:-1])] + input1_axes_shape[-1:]
            if len(op.y_out_axes) > 2:
                reshape_input_needed = True
                reshape_output_needed = True
                input2_axes_shape = input2_axes_shape[:1] + [np.prod(input2_axes_shape[1:])]

        # reshape input
        if reshape_input_needed:
            input1_op = PyngReshape(
                self.computation.lookup_cpp_op(input1),
                input1_axes_order,
                input1_axes_shape)
            input2_op = PyngReshape(
                self.computation.lookup_cpp_op(input2),
                input2_axes_order,
                input2_axes_shape)
        else:
            input1_op = self.computation.lookup_cpp_op(input1)
            input2_op = self.computation.lookup_cpp_op(input2)

        ngraph_op = PyngDot(input1_op, input2_op,
                            reduction_axes_count)
        # reshape output
        if reshape_output_needed:
            ngraph_op = PyngReshape(
                ngraph_op,
                list(range(0, 4 - 2*reduction_axes_count)),
                list(op.x_out_axes.lengths) + list(op.y_out_axes.lengths))
        # reshape output
        self.computation.register_cpp_op(op, ngraph_op)

    @visit.on_type(LogOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        ngraph_cpp_log_op = PyngLog(self.computation.lookup_cpp_op(input))
        self.computation.register_cpp_op(op, ngraph_cpp_log_op)

    @visit.on_type(ExpOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        ngraph_cpp_exp_op = PyngExp(self.computation.lookup_cpp_op(input))
        self.computation.register_cpp_op(op, ngraph_cpp_exp_op)

    @visit.on_type(Greater)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_greater_op = PyngGreater(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        # convert the element back from bool to float type
        element_result_type = Type.f32
        greater_result_op = PyngConvert(ngraph_cpp_greater_op, element_result_type)
        self.computation.register_cpp_op(op, greater_result_op)

    @visit.on_type(GreaterEqual)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_greater_op = PyngGreaterEq(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        # convert the element back from bool to float type
        element_result_type = Type.f32
        greater_result_op = PyngConvert(ngraph_cpp_greater_op, element_result_type)
        self.computation.register_cpp_op(op, greater_result_op)

    @visit.on_type(Less)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_less_op = PyngLess(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        # convert the element back from bool to float type
        element_result_type = Type.f32
        less_result_op = PyngConvert(ngraph_cpp_less_op, element_result_type)
        self.computation.register_cpp_op(op, less_result_op)

    @visit.on_type(Equal)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_equal_op = PyngEqual(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        # convert the element back from bool to float type
        element_result_type = Type.f32
        equal_result_op = PyngConvert(ngraph_cpp_equal_op, element_result_type)
        self.computation.register_cpp_op(op, equal_result_op)

    @visit.on_type(NotEqual)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_notequal_op = PyngNotEqual(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        # convert the element back from bool to float type
        element_result_type = Type.f32
        notequal_result_op = PyngConvert(ngraph_cpp_notequal_op, element_result_type)
        self.computation.register_cpp_op(op, notequal_result_op)

    @visit.on_type(Sum)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        if isinstance(self.np_reduction_axis(op), tuple):
            axis_set = self.np_reduction_axis(op)
        else:
            axis_set = tuple()
            axis_set += (self.np_reduction_axis(op),)

        ngraph_cpp_sum_op = PyngSum(
            self.computation.lookup_cpp_op(input),
            set(axis_set))
        self.computation.register_cpp_op(op, ngraph_cpp_sum_op)

    @visit.on_type(Maximum)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_maximum_op = PyngMaximum(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        self.computation.register_cpp_op(op, ngraph_cpp_maximum_op)

    @visit.on_type(Minimum)
    def visit(self, op, input1, input2):
        self.computation.set_op_rank(op)
        ngraph_cpp_minimum_op = PyngMinimum(
            self.computation.lookup_cpp_op(input1),
            self.computation.lookup_cpp_op(input2))
        self.computation.register_cpp_op(op, ngraph_cpp_minimum_op)

    @visit.on_type(ReorderAxes)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        axis_order = []
        reorder_axes = list(op.axes.lengths)
        reorder_axes_names = op.axes.names
        input_axes_names = op.args[0].axes.names

        # determine the axis order for the reshape
        for input_axis_name in input_axes_names:
            index = reorder_axes_names.index(input_axis_name)
            axis_order.append(index)
        ngraph_input = self.computation.lookup_cpp_op(op.args[0])
        # print(ngraph_input.get_output_shape(0))
        ngraph_cpp_reorder_op = PyngReshape(
            ngraph_input,
            axis_order,
            reorder_axes)
        self.computation.register_cpp_op(op, ngraph_cpp_reorder_op)

    @visit.on_type(OneHotOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        onehot_shape = list(op.axes.lengths)
        one_hot_axis = (op.axes).index(op.axis)
        ngraph_cpp_onehot_op = PyngOneHot(
            self.computation.lookup_cpp_op(op.args[0]),
            onehot_shape,
            one_hot_axis)
        self.computation.register_cpp_op(op, ngraph_cpp_onehot_op)

    @visit.on_type(NegativeOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        ngraph_cpp_neg_op = PyngNegative(
            self.computation.lookup_cpp_op(input))
        self.computation.register_cpp_op(op, ngraph_cpp_neg_op)

    @visit.on_type(Prod)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        # Define the reduction function handle
        element_type = Type.f32
        shape = []
        f_a = Parameter(element_type, shape)
        f_b = Parameter(element_type, shape)
        ngraph_cpp_mul_op = PyngMultiply(f_a, f_b)
        fn = Function([ngraph_cpp_mul_op], [f_a, f_b], self.transformer.get_function_name())

        # define the reduction op with the above defined Function handle
        if isinstance(self.np_reduction_axis(op), tuple):
            axis_set = self.np_reduction_axis(op)
        else:
            axis_set = tuple()
            axis_set += (self.np_reduction_axis(op),)
        g_a = self.computation.lookup_cpp_op(input)
        const_prod_default_value = [1.]
        g_b = Constant(Type.f32, [], const_prod_default_value)

        self.computation.register_cpp_op(op, PyngReduce(g_a, g_b, fn, set(axis_set)))

    @visit.on_type(ReciprocalOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        input_axes = list(input.axes.lengths)
        constant_op = Constant(Type.f32, input_axes, [1])
        ngraph_cpp_reciprocal_op = constant_op \
            / self.computation.lookup_cpp_op(input)
        self.computation.register_cpp_op(op, ngraph_cpp_reciprocal_op)

    @visit.on_type(TensorSizeOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        # TODO - is treating TensorSizeOp as constants, okay?
        # Construct constant list with number of elements = reduction axes size
        constant_tensor = [op.reduction_axes.size]
        constant_op = Constant(Type.f32,
                               [], constant_tensor)
        self.computation.register_cpp_op(op, constant_op)

    @visit.on_type(MapRolesOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        # TODO - made it as workaround, need to check if this acceptable ?
        self.computation.register_cpp_op(
            op, self.computation.lookup_cpp_op(op.args[0]))

    @visit.on_type(Max)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        # Define the reduction function handle
        element_type = Type.f32
        shape = []
        f_a = Parameter(element_type, shape)
        f_b = Parameter(element_type, shape)
        ngraph_cpp_min_op = PyngMaximum(f_a, f_b)
        fn = Function([ngraph_cpp_min_op], [f_a, f_b], self.transformer.get_function_name())

        # define the reduction op with the above defined Function handle
        if isinstance(self.np_reduction_axis(op), tuple):
            axis_set = self.np_reduction_axis(op)
        else:
            axis_set = tuple()
            axis_set += (self.np_reduction_axis(op),)
        g_a = self.computation.lookup_cpp_op(input)
        const_max_default_value = [float('-inf')]
        g_b = Constant(Type.f32, [], const_max_default_value)
        self.computation.register_cpp_op(op, PyngReduce(g_a, g_b, fn, set(axis_set)))

    @visit.on_type(SequentialOp)
    def visit(self, op):
        self.computation.set_op_rank(op)
        # Legal child patterns
        # 1. (AssignOp,)+, (~(SequentialOp|ParallelOp))
        # 2. ParallelOp, (~(AssignOp|SequentialOp|ParallelOp))
        # 3. SequentialOp, (~(AssignOp|SequentialOp|ParallelOp))

        # Output node is the last child op
        self.computation.register_cpp_op(
            op, self.computation.lookup_cpp_op(op.ops[-1]))

    @visit.on_type(ParallelOp)
    def visit(self, op):
        self.computation.set_op_rank(op)
        # Legal child pattern
        # 1. (AssignOp,)+
        # 2. (SequentialOp,)+ where SequentialOp = (AssignOp,)+

        # ParallelOp has no output node

    @visit.on_type(AssignOp)
    def visit(self, op, lhs, rhs):
        self.computation.set_op_rank(op)
        variable = lhs.tensor
        if variable not in self.computation.variables_cpp_op:
            self.computation.variables_cpp_op[variable] = \
                (self.computation.scopemark[op.tensor], rhs)
            self.computation.register_cpp_op(
                op, self.computation.lookup_cpp_op(rhs))
        else:
            raise RuntimeError("Variable updated more than once!")

    @visit.on_type(ExpandDims)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        op_element_type = self.computation.lookup_cpp_op(x)
        axis_set = set()
        axis_set.add(op.dim)
        self.computation.register_cpp_op(op, PyngBroadcast(op_element_type,
                                         list(op.axes.lengths), axis_set))

    """
    /// brief Constructs a batched convolution operation.
    ///
    /// param data_batch The node producing the input data batch tensor.
    /// param filters The node producing the filters tensor.
    /// param window_movement_strides The window movement strides.
    /// param window_dilation_strides The window dilation strides.
    /// param padding_below The padding-below sizes.
    /// param padding_above The padding-above sizes.
    /// param data_dilation_strides The data dilation strides.
    """
    @visit.on_type(ConvolutionOp)
    def visit(self, op, *args):
        self.computation.set_op_rank(op)
        # op.args[0] : inputs
        # op.args[1] : filters
        # op.args[2] (optional): bias
        # op.conv_params
        # op.channel_axes
        # op.spatial_axes
        if len(args) == 2:
            inputs = args[0]
            filters = args[1]
        else:
            raise RuntimeError("Not Implemented: Convolution with bias")

        """
        {'K': 16, 'T': 1, 'R': 5, 'S': 5, 'str_d': 1, 'pad_d': 0, 'dil_d': 1,
        'str_h': 1, 'pad_h': 0, 'dil_h': 1, 'str_w': 1, 'pad_w': 0, 'dil_w': 1}
        """
        """
        print(inputs.axes)
        print(op.axes)
        print(filters.axes)
        """
        # print(op_element_type.get_output_shape(0))
        reordered = PyngReshape(self.computation.lookup_cpp_op(inputs), [4, 0, 1, 2, 3],
                                [inputs.axes[4].length, inputs.axes[0].length,
                                inputs.axes[2].length, inputs.axes[3].length])
        filters_reordered = PyngReshape(self.computation.lookup_cpp_op(filters), [4, 0, 1, 2, 3],
                                        [filters.axes[4].length, filters.axes[0].length,
                                         filters.axes[2].length, filters.axes[3].length])
        ngraph_conv = PyngConvolution(
            reordered,
            filters_reordered,
            [op.conv_params['str_h'], op.conv_params['str_w']],
            [op.conv_params['dil_h'], op.conv_params['dil_w']],
            [op.conv_params['pad_h'], op.conv_params['pad_w']],
            [op.conv_params['pad_h'], op.conv_params['pad_w']],
            [1, 1])

        ordered = PyngReshape(ngraph_conv, [1, 2, 3, 0],    #NCHW --> CDHWN
                              list(op.axes.lengths))

        self.computation.register_cpp_op(op, ordered)

    """
    /// brief Constructs a batched-convolution data batch-backprop operation.
    ///
    /// param data_batch_shape The shape of the data batch from forward-prop.
    /// param filters The node producing the filters from forward-prop.
    /// param output_delta The node producing output delta.
    /// param window_movement_strides_forward The window movement strides from forward-prop.
    /// param window_dilation_strides_forward The window dilation strides from forward-prop.
    /// param padding_below_forward The padding-below sizes from forward-prop.
    /// param padding_above_forward The padding-above sizes from forward-prop.
    /// param data_dilation_strides_forward The data dilation strides from forward-prop.
    ConvolutionBackpropData(const Shape& data_batch_shape,
                            const std::shared_ptr<Node>& filters,
                            const std::shared_ptr<Node>& output_delta,
                            const Strides& window_movement_strides_forward,
                            const Strides& window_dilation_strides_forward,
                            const CoordinateDiff& padding_below_forward,
                            const CoordinateDiff& padding_above_forward,
                            const Strides& data_dilation_strides_forward);
    """
    @visit.on_type(bprop_conv)
    def visit(self, op, *args):
        self.computation.set_op_rank(op)
        # op.args[0] : delta
        # op.args[1] : filters
        # op.fprop.args[0] : fprop data batch shape
        # op.fprop.conv_params : forward params
        delta = args[0]
        filters = args[1]
        data = op.fprop.args[0]
        conv_params = op.fprop.conv_params
        """
        print(delta.axes)
        print(filters.axes)
        print(data.axes)
        print(conv_params)
        """
        delta_reordered = PyngReshape(self.computation.lookup_cpp_op(delta), [4, 0, 1, 2, 3],
                                      [delta.axes[4].length, delta.axes[0].length,
                                       delta.axes[2].length, delta.axes[3].length])
        filters_reordered = PyngReshape(self.computation.lookup_cpp_op(filters), [4, 0, 1, 2, 3],
                                        [filters.axes[4].length, filters.axes[0].length,
                                         filters.axes[2].length, filters.axes[3].length])
        ngraph_bprop_conv = PyngConvolutionBackpropData(
            [data.axes[4].length, data.axes[0].length,
             data.axes[2].length, data.axes[3].length],
            filters_reordered,
            delta_reordered,
            [conv_params['str_h'], conv_params['str_w']],
            [conv_params['dil_h'], conv_params['dil_w']],
            [conv_params['pad_h'], conv_params['pad_w']],
            [conv_params['pad_h'], conv_params['pad_w']],
            [1, 1])

        ordered = PyngReshape(ngraph_bprop_conv, [1, 2, 3, 0],
                              list(op.axes.lengths))

        self.computation.register_cpp_op(op, ordered)

    """
    /// brief Constructs a batched-convolution filter-backprop operation.
    ///
    /// param data_batch The tensor producing the data batch from forward-prop.
    /// param filters_shape The shape of the filters from forward-prop.
    /// param output_delta The node producing output delta.
    /// param window_movement_strides_forward The window movement strides from forward-prop.
    /// param window_dilation_strides_forward The window dilation strides from forward-prop.
    /// param padding_below_forward The padding-below sizes from forward-prop.
    /// param padding_above_forward The padding-above sizes from forward-prop.
    /// param data_dilation_strides_forward The data dilation strides from forward-prop.
    ConvolutionBackpropFilters(const std::shared_ptr<Node>& data_batch,
                                const Shape& filters_shape,
                                const std::shared_ptr<Node>& output_delta,
                                const Strides& window_movement_strides_forward,
                                const Strides& window_dilation_strides_forward,
                                const CoordinateDiff& padding_below_forward,
                                const CoordinateDiff& padding_above_forward,
                                const Strides& data_dilation_strides_forward);
    """
    @visit.on_type(update_conv)
    def visit(self, op, *args):
        self.computation.set_op_rank(op)
        # op.args[0] : delta
        # op.args[1] : data batch
        # op.args[2] (optional) : dbias
        # op.fprop.args[0] : data batch
        # op.fprop.args[1] : filters
        # op.fprop.conv_params : forward params
        delta = args[0]
        data = args[1]
        filters = op.fprop.args[1]
        conv_params = op.fprop.conv_params
        """
        print(delta.axes)
        print(filters.axes)
        print(data.axes)
        print(conv_params)
        """
        data_reordered = PyngReshape(self.computation.lookup_cpp_op(data), [4, 0, 1, 2, 3],
                                     [data.axes[4].length, data.axes[0].length,
                                      data.axes[2].length, data.axes[3].length])
        delta_reordered = PyngReshape(self.computation.lookup_cpp_op(delta), [4, 0, 1, 2, 3],
                                      [delta.axes[4].length, delta.axes[0].length,
                                      delta.axes[2].length, delta.axes[3].length])

        ngraph_update_conv = PyngConvolutionBackpropFilters(
            data_reordered,
            [filters.axes[4].length, filters.axes[0].length,
             filters.axes[2].length, filters.axes[3].length],
            delta_reordered,
            [conv_params['str_h'], conv_params['str_w']],
            [conv_params['dil_h'], conv_params['dil_w']],
            [conv_params['pad_h'], conv_params['pad_w']],
            [conv_params['pad_h'], conv_params['pad_w']],
            [1, 1])

        ordered = PyngReshape(ngraph_update_conv, [1, 2, 3, 0],
                              list(op.axes.lengths))

        self.computation.register_cpp_op(op, ordered)

    """
    /// brief Constructs a batched max pooling operation.
    ///
    /// param arg The node producing the input data batch tensor.
    /// param window_shape The window shape.
    /// param window_movement_strides The window movement strides.
    /// param padding_below The below-padding shape.
    /// param padding_above The above-padding shape.

    /// brief Constructs a batched average pooling operation.
    ///
    /// param arg The node producing the input data batch tensor.
    /// param window_shape The window shape.
    /// param window_movement_strides The window movement strides.
    /// param padding_below The below-padding shape.
    /// param padding_above The above-padding shape.
    """
    @visit.on_type(PoolingOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        # op.args[0] : inputs
        # op.pool_params
        # op.channel_axes
        # op.spatial_axes
        if 'max' == op.pool_params['op']:
            """
            print(op.pool_params)
            print(inputs.axes)
            print(op.axes)
            """
            reordered = PyngReshape(self.computation.lookup_cpp_op(inputs), [4, 0, 1, 2, 3],
                                    [inputs.axes[4].length, inputs.axes[0].length,
                                    inputs.axes[2].length, inputs.axes[3].length])
            ngraph_pool = PyngMaxPool(reordered,
                                      [op.pool_params['R'],
                                          op.pool_params['S']],
                                      [op.pool_params['str_h'],
                                          op.pool_params['str_w']],
                                      [op.pool_params['pad_h'],
                                          op.pool_params['pad_w']],
                                      [op.pool_params['pad_h'],
                                          op.pool_params['pad_w']])
            ordered = PyngReshape(ngraph_pool, [1, 2, 3, 0],
                                  list(op.axes.lengths))

            self.computation.register_cpp_op(op, ordered)
        elif 'avg' == op.pool_params['op']:
            reordered = PyngReshape(self.computation.lookup_cpp_op(inputs), [4, 0, 1, 2, 3],
                                    [inputs.axes[4].length, inputs.axes[0].length,
                                    inputs.axes[2].length, inputs.axes[3].length])
            ngraph_pool = PyngAvgPool(reordered,
                                      [op.pool_params['R'],
                                          op.pool_params['S']],
                                      [op.pool_params['str_h'],
                                          op.pool_params['str_w']],
                                      [op.pool_params['pad_h'],
                                          op.pool_params['pad_w']],
                                      [op.pool_params['pad_h'],
                                          op.pool_params['pad_w']])
            """
            print(list(op.axes.lengths))
            print(ngraph_pool.get_output_shape(0))
            """
            ordered = PyngReshape(ngraph_pool, [1, 2, 3, 0],
                                  list(op.axes.lengths))

            self.computation.register_cpp_op(op, ordered)
        else:
            raise RuntimeError("Unsupported pooling type: " + op.pool_params['op'])

    @visit.on_type(BpropPoolOp)
    def visit(self, op, delta):
        self.computation.set_op_rank(op)
        # op.args[0] : delta
        # op.fprop
        # op.inputs
        if 'max' == op.pool_params['op']:
            """
            MaxPoolBackprop(const std::shared_ptr<Node>& arg_forward,
                    const std::shared_ptr<Node>& delta,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    const std::shared_ptr<op::MaxPool>& forward_op = nullptr);
            """
            """
            print(delta.axes)
            print(op.inputs.axes)
            print(op.axes)
            """
            inputs = op.inputs
            reordered = PyngReshape(self.computation.lookup_cpp_op(inputs), [4, 0, 1, 2, 3],
                                    [inputs.axes[4].length, inputs.axes[0].length,
                                    inputs.axes[2].length, inputs.axes[3].length])

            red_delta = PyngReshape(self.computation.lookup_cpp_op(delta), [4, 0, 1, 2, 3],
                                    [delta.axes[4].length, delta.axes[0].length,
                                     delta.axes[2].length, delta.axes[3].length])
            ngraph_fprop = self.computation.lookup_cpp_op(op.fprop).get_input_op(0)
            """
            print(red_delta.get_output_shape(0))
            print(ngraph_fprop.get_output_shape(0))
            """
            ngraph_pool = PyngMaxPoolBackprop(reordered,
                                              red_delta,
                                              [op.fprop.pool_params['R'],
                                                  op.fprop.pool_params['S']],
                                              [op.fprop.pool_params['str_h'],
                                                  op.fprop.pool_params['str_w']],
                                              [op.fprop.pool_params['pad_h'],
                                                  op.fprop.pool_params['pad_w']],
                                              [op.fprop.pool_params['pad_h'],
                                                  op.fprop.pool_params['pad_w']],
                                              ngraph_fprop)
            ordered = PyngReshape(ngraph_pool, [1, 2, 3, 0],
                                  list(op.axes.lengths))

            self.computation.register_cpp_op(op, ordered)
        elif 'avg' == op.pool_params['op']:
            """
            AvgPoolBackprop(const Shape& forward_arg_shape,
                const std::shared_ptr<Node>& delta,
                const Shape& window_shape,
                const Strides& window_movement_strides,
                const Shape& padding_below,
                const Shape& padding_above);
            """
            """
            print(delta.axes)
            print(op.inputs.axes)
            print(op.axes)
            """
            red_delta = PyngReshape(self.computation.lookup_cpp_op(delta), [4, 0, 1, 2, 3],
                                    [delta.axes[4].length, delta.axes[0].length,
                                    delta.axes[2].length, delta.axes[3].length])
            """
            print(red_delta.get_output_shape(0))
            print(ngraph_fprop.get_output_shape(0))
            """
            inputs = op.inputs
            ngraph_pool = PyngAvgPoolBackprop([inputs.axes[4].length, inputs.axes[0].length,
                                              inputs.axes[2].length, inputs.axes[3].length],
                                              red_delta,
                                              [op.fprop.pool_params['R'],
                                                  op.fprop.pool_params['S']],
                                              [op.fprop.pool_params['str_h'],
                                                  op.fprop.pool_params['str_w']],
                                              [op.fprop.pool_params['pad_h'],
                                                  op.fprop.pool_params['pad_w']],
                                              [op.fprop.pool_params['pad_h'],
                                                  op.fprop.pool_params['pad_w']])
            ordered = PyngReshape(ngraph_pool, [1, 2, 3, 0],
                                  list(op.axes.lengths))

            self.computation.register_cpp_op(op, ordered)
        else:
            raise RuntimeError("Unsupported pooling type: " + op.pool_params['op'])

    @visit.on_type(TensorSliceOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        # op.args[0] : x
        # op.slices
        lowers = []
        uppers = []
        strides = []
        axes_to_remove = []
        for axis, s in zip(x.axes, op.slices):
            if isinstance(s, int):
                lowers.append(s)
                uppers.append(s + 1)
                strides.append(1)
                axes_to_remove.append(axis)
            else:
                if s.start is None:
                    lowers.append(0)
                else:
                    lowers.append(s.start)
                if s.step is None:
                    strides.append(1)
                else:
                    strides.append(s.step)
                if s.stop is None:
                    uppers.append(axis.length)
                else:
                    uppers.append(s.stop)
        op_element_type = self.computation.lookup_cpp_op(x)
        """
        print("TensorSliceOp")
        print(x.axes)
        print(op.axes)
        print(op_element_type.get_output_shape(0))
        print(lowers)
        print(uppers)
        print(strides)
        """
        ngraph_sliced = PyngSlice(op_element_type, lowers, uppers, strides)
        if axes_to_remove:
            ngraph_sliced = PyngReshape(ngraph_sliced,
                                        list(range(0, len(x.axes))),
                                        list(op.axes.lengths))

        self.computation.register_cpp_op(op, ngraph_sliced)

    @visit.on_type(SqrtOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_x = self.computation.lookup_cpp_op(x)
        ngraph_sqrt = PyngSqrt(ngraph_x)
        self.computation.register_cpp_op(op, ngraph_sqrt)

    @visit.on_type(SquareOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_x = self.computation.lookup_cpp_op(x)
        self.computation.register_cpp_op(op, ngraph_x * ngraph_x)

    @visit.on_type(Flatten)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_flatten = PyngReshape(self.computation.lookup_cpp_op(x),
                                     list(range(0, len(x.axes))),
                                     list(op.axes.lengths))
        self.computation.register_cpp_op(op, ngraph_flatten)

    @visit.on_type(Unflatten)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        self.computation.set_op_rank(op)
        ngraph_unflatten = PyngReshape(self.computation.lookup_cpp_op(x),
                                       list(range(0, len(x.axes))),
                                       list(op.axes.lengths))
        self.computation.register_cpp_op(op, ngraph_unflatten)

    @visit.on_type(ContiguousOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_x = self.computation.lookup_cpp_op(x)
        self.computation.register_cpp_op(op, ngraph_x)
