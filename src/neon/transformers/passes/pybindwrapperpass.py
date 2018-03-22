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

from __future__ import division
from neon.transformers.passes.passes import PeepholeGraphPass
from neon.util.generics import generic_method
from neon.op_graph.op_graph import Op, Add, AssignableTensorOp, AssignOp, AxesCastOp, \
    BroadcastOp, ContiguousOp, Divide, DotOp, Equal, ExpandDims, ExpOp, Flatten, Greater, \
    GreaterEqual, Less, LogOp, MapRolesOp, Max, Maximum, Minimum, Multiply, NegativeOp, \
    NotEqual, OneHotOp, ParallelOp, Prod, ReciprocalOp, ReductionOp, ReorderAxes, \
    SequentialOp, SqrtOp, SquareOp, Subtract, Sum, TensorSliceOp, TensorSizeOp, TensorValueOp, \
    Unflatten, Fill
from neon.op_graph.batchnorm import BatchnormCommonOp, BatchnormBpropCommonOp, \
    BatchnormOutputOp, BatchnormMeanOp, BatchnormVarOp, \
    BatchnormBpropDataOp, BatchnormBpropGammaOp, BatchnormBpropBetaOp
from neon.op_graph.relu import ReluOp, ReluBpropOp
from neon.op_graph.pooling import PoolingOp, BpropPoolOp
from neon.op_graph.convolution import ConvolutionOp, bprop_conv, update_conv
import numpy as np

from ngraph.impl import Type
from ngraph.impl import Shape
from ngraph.impl import AxisSet
from ngraph.impl import AxisVector
from ngraph.impl import Strides
from ngraph.impl import CoordinateDiff
from ngraph.impl import Coordinate
from ngraph.impl.op import Parameter
from ngraph.impl.op import AvgPool as PyngAvgPool
from ngraph.impl.op import AvgPoolBackprop as PyngAvgPoolBackprop
from ngraph.impl.op import Broadcast as PyngBroadcast
from ngraph.impl.op import Constant
from ngraph.impl.op import Convert as PyngConvert
from ngraph.impl.op import Convolution as PyngConvolution
from ngraph.impl.op import ConvolutionBackpropData as PyngConvolutionBackpropData
from ngraph.impl.op import ConvolutionBackpropFilters as PyngConvolutionBackpropFilters
from ngraph.impl.op import Dot as PyngDot
from ngraph.impl.op import Equal as PyngEqual
from ngraph.impl.op import Exp as PyngExp
from ngraph.impl.op import Greater as PyngGreater
from ngraph.impl.op import GreaterEq as PyngGreaterEq
from ngraph.impl.op import Less as PyngLess
from ngraph.impl.op import Log as PyngLog
from ngraph.impl.op import Maximum as PyngMaximum
from ngraph.impl.op import MaxPool as PyngMaxPool
from ngraph.impl.op import MaxPoolBackprop as PyngMaxPoolBackprop
from ngraph.impl.op import Minimum as PyngMinimum
from ngraph.impl.op import Negative as PyngNegative
from ngraph.impl.op import NotEqual as PyngNotEqual
from ngraph.impl.op import OneHot as PyngOneHot
from ngraph.impl.op import Reshape as PyngReshape
from ngraph.impl.op import Slice as PyngSlice
from ngraph.impl.op import Sqrt as PyngSqrt
from ngraph.impl.op import Sum as PyngSum
from ngraph.impl.op import BatchNorm as PyngBatchNorm
from ngraph.impl.op import BatchNormBackprop as PyngBatchNormBackprop
from ngraph.impl.op import GetOutputElement as PyngGetOutputElement
from ngraph.impl.op import Max as PyngMax
from ngraph.impl.op import Product as PyngProduct
from ngraph.impl.op import Relu as PyngRelu
from ngraph.impl.op import ReluBackprop as PyngReluBackprop


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

        def update_scopemark(op, scope):
            if op not in self.computation.scopemark:
                self.computation.scopemark[op] = scope
            else:
                old_scope = self.computation.scopemark[op]
                # if the new scope has a longer path, update the scope
                if scope.count('/') > old_scope.count('/'):
                    self.computation.scopemark[op] = scope

        def visit_pre_order(scope, op):
            visited = set()
            nodes_to_visit = [(op, scope)]

            while nodes_to_visit:
                op, scope = nodes_to_visit.pop()
                if op in visited:
                    continue
                visited.add(op)
                if isinstance(op, TensorValueOp):
                    update_scopemark(op, scope)
                elif isinstance(op, SequentialOp):
                    childscope = new_seq_scope(scope)
                    children = op.ops
                    update_scopemark(op, scope)
                    for child in children:
                        nodes_to_visit.append((child, childscope))
                elif isinstance(op, ParallelOp):
                    childscope = new_par_scope(scope)
                    children = op.control_deps
                    update_scopemark(op, scope)
                    for child in children:
                        nodes_to_visit.append((child, childscope))
                else:
                    tensor_op = op.tensor
                    childscope = scope
                    children = tensor_op.args
                    update_scopemark(op, scope)
                    for child in children:
                        nodes_to_visit.append((child, childscope))

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

    def binary_op(self, op, x, y, is_logical=False):

        def pyng_binary_op(op, x, y):
            if isinstance(op, Add):
                return x + y
            elif isinstance(op, Divide):
                return x / y
            elif isinstance(op, Multiply):
                return x * y
            elif isinstance(op, Subtract):
                return x - y
            elif isinstance(op, Greater):
                return PyngGreater(x, y)
            elif isinstance(op, GreaterEqual):
                return PyngGreaterEq(x, y)
            elif isinstance(op, Less):
                return PyngLess(x, y)
            elif isinstance(op, Equal):
                return PyngEqual(x, y)
            elif isinstance(op, NotEqual):
                return PyngNotEqual(x, y)
            elif isinstance(op, Maximum):
                return PyngMaximum(x, y)
            elif isinstance(op, Minimum):
                return PyngMinimum(x, y)

        self.computation.set_op_rank(op)
        ngraph_cpp_op = pyng_binary_op(op, self.computation.lookup_cpp_op(x),
                                       self.computation.lookup_cpp_op(y))
        if is_logical:
            element_result_type = Type.f32
            ngraph_cpp_op = PyngConvert(ngraph_cpp_op, element_result_type)
        self.computation.register_cpp_op(op, ngraph_cpp_op)

    def unary_op(self, op, x):

        def pyng_unary_op(op, x):
            if isinstance(op, LogOp):
                return PyngLog(x)
            elif isinstance(op, ExpOp):
                return PyngExp(x)
            elif isinstance(op, SquareOp):
                return x * x
            elif isinstance(op, SqrtOp):
                return PyngSqrt(x)
            elif isinstance(op, NegativeOp):
                return PyngNegative(x)

        self.computation.set_op_rank(op)
        ngraph_cpp_op = pyng_unary_op(op, self.computation.lookup_cpp_op(x))
        self.computation.register_cpp_op(op, ngraph_cpp_op)

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        self.computation.set_op_rank(op)
        raise RuntimeError("Unsupported op " + str(type(op)))

    @visit.on_type(Add)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(Divide)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(Multiply)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(Subtract)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(AxesCastOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        op_element_type = self.computation.lookup_cpp_op(x)
        self.computation.register_cpp_op(op, op_element_type, set_name=False)

    @visit.on_type(BroadcastOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        axis_set = set()
        op_element_type = self.computation.lookup_cpp_op(op.args[0])

        # build axis_set
        broadcast_axes = op.axes.names
        broadcast_args_axes = op.args[0].axes.names

        for pos, axis in enumerate(broadcast_axes):
            if axis not in broadcast_args_axes:
                axis_set.add(pos)

        self.computation.register_cpp_op(
            op, PyngBroadcast(op_element_type, Shape(list(op.axes.lengths)),
                              AxisSet(axis_set)))

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
                                       Shape(list(tensor.axes.lengths)),
                                       list(self.flatten(tensor.const.tolist())))
                self.computation.register_cpp_op(tensor, constant_op)
            else:
                op_element_type = Parameter(Type.f32, Shape(list(tensor.axes.lengths)))
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
                                       Shape(list(op.axes.lengths)),
                                       list(self.flatten(op.const.tolist())))

                self.computation.register_cpp_op(op, constant_op)
            else:
                op_element_type = Parameter(Type.f32, Shape(list(op.axes.lengths)))
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
            if len(op.y_out_axes) > 1:
                reshape_input_needed = True
                reshape_output_needed = True
                input2_axes_shape = input2_axes_shape[:1] + [np.prod(input2_axes_shape[1:])]

        # reshape input
        if reshape_input_needed:
            input1_op = PyngReshape(
                self.computation.lookup_cpp_op(input1),
                AxisVector(input1_axes_order),
                Shape(input1_axes_shape))
            input2_op = PyngReshape(
                self.computation.lookup_cpp_op(input2),
                AxisVector(input2_axes_order),
                Shape(input2_axes_shape))
        else:
            input1_op = self.computation.lookup_cpp_op(input1)
            input2_op = self.computation.lookup_cpp_op(input2)

        ngraph_op = PyngDot(input1_op, input2_op,
                            reduction_axes_count)
        # reshape output
        if reshape_output_needed:
            ngraph_op = PyngReshape(
                ngraph_op,
                AxisVector(list(range(0, 4 - 2 * reduction_axes_count))),
                Shape(list(op.x_out_axes.lengths) + list(op.y_out_axes.lengths)))
        # reshape output
        self.computation.register_cpp_op(op, ngraph_op)

    @visit.on_type(LogOp)
    def visit(self, op, input):
        self.unary_op(op, input)

    @visit.on_type(ExpOp)
    def visit(self, op, input):
        self.unary_op(op, input)

    @visit.on_type(Greater)
    def visit(self, op, x, y):
        self.binary_op(op, x, y, is_logical=True)

    @visit.on_type(GreaterEqual)
    def visit(self, op, x, y):
        self.binary_op(op, x, y, is_logical=True)

    @visit.on_type(Less)
    def visit(self, op, x, y):
        self.binary_op(op, x, y, is_logical=True)

    @visit.on_type(Equal)
    def visit(self, op, x, y):
        self.binary_op(op, x, y, is_logical=True)

    @visit.on_type(NotEqual)
    def visit(self, op, x, y):
        self.binary_op(op, x, y, is_logical=True)

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
            AxisSet(set(axis_set)))
        self.computation.register_cpp_op(op, ngraph_cpp_sum_op)

    @visit.on_type(Maximum)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(Minimum)
    def visit(self, op, x, y):
        self.binary_op(op, x, y)

    @visit.on_type(ReorderAxes)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        axis_order = []
        reorder_axes = list(op.axes.lengths)
        reorder_axes_names = op.axes.names
        input_axes_names = op.args[0].axes.names

        # determine the axis order for the reshape
        for reorder_axis_name in reorder_axes_names:
            index = input_axes_names.index(reorder_axis_name)
            axis_order.append(index)
        ngraph_input = self.computation.lookup_cpp_op(op.args[0])
        # print(ngraph_input.get_output_shape(0))
        ngraph_cpp_reorder_op = PyngReshape(
            ngraph_input,
            AxisVector(axis_order),
            Shape(reorder_axes))
        self.computation.register_cpp_op(op, ngraph_cpp_reorder_op)

    @visit.on_type(OneHotOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        onehot_shape = list(op.axes.lengths)
        one_hot_axis = (op.axes).index(op.axis)
        ngraph_cpp_onehot_op = PyngOneHot(
            self.computation.lookup_cpp_op(op.args[0]),
            Shape(onehot_shape),
            one_hot_axis)
        self.computation.register_cpp_op(op, ngraph_cpp_onehot_op)

    @visit.on_type(NegativeOp)
    def visit(self, op, x):
        self.unary_op(op, x)

    @visit.on_type(Prod)
    def visit(self, op, input):
        self.computation.set_op_rank(op)

        if isinstance(self.np_reduction_axis(op), tuple):
            axis_set = self.np_reduction_axis(op)
        else:
            axis_set = tuple()
            axis_set += (self.np_reduction_axis(op),)
        ngraph_input = self.computation.lookup_cpp_op(input)
        self.computation.register_cpp_op(op, PyngProduct(ngraph_input, AxisSet(set(axis_set))))

    @visit.on_type(ReciprocalOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        input_axes = list(input.axes.lengths)
        constant_op = Constant(Type.f32, Shape(input_axes), [1])
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
                               Shape([]), constant_tensor)
        self.computation.register_cpp_op(op, constant_op)

    @visit.on_type(MapRolesOp)
    def visit(self, op, input):
        self.computation.set_op_rank(op)
        # TODO - made it as workaround, need to check if this acceptable ?
        self.computation.register_cpp_op(
            op, self.computation.lookup_cpp_op(op.args[0]), set_name=False)

    @visit.on_type(Max)
    def visit(self, op, input):
        self.computation.set_op_rank(op)

        if isinstance(self.np_reduction_axis(op), tuple):
            axis_set = self.np_reduction_axis(op)
        else:
            axis_set = tuple()
            axis_set += (self.np_reduction_axis(op),)
        ngraph_input = self.computation.lookup_cpp_op(input)
        self.computation.register_cpp_op(op, PyngMax(ngraph_input, AxisSet(set(axis_set))))

    @visit.on_type(SequentialOp)
    def visit(self, op):
        self.computation.set_op_rank(op)
        # Legal child patterns
        # 1. (AssignOp,)+, (~(SequentialOp|ParallelOp))
        # 2. ParallelOp, (~(AssignOp|SequentialOp|ParallelOp))
        # 3. SequentialOp, (~(AssignOp|SequentialOp|ParallelOp))

        # Output node is the last child op
        self.computation.register_cpp_op(
            op, self.computation.lookup_cpp_op(op.ops[-1]), set_name=False)

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
                op, self.computation.lookup_cpp_op(rhs), set_name=False)
        else:
            raise RuntimeError("Variable updated more than once!")

    @visit.on_type(Fill)
    def visit(self, op, tensor):
        self.computation.set_op_rank(op)
        variable = tensor.tensor
        list_size = 1
        for x in list(tensor.axes.lengths):
            list_size *= x
        constant_list = [op.scalar] * list_size
        ngraph_constant_op = Constant(Type.f32,
                                      Shape(list(tensor.axes.lengths)),
                                      constant_list)
        if variable not in self.computation.variables_cpp_op:
            # treat 'op' as the rhs of assignment for forwarding and lookup purposes
            self.computation.variables_cpp_op[variable] = \
                (self.computation.scopemark[op.tensor], op)
            self.computation.register_cpp_op(
                op, ngraph_constant_op, set_name=False)
        else:
            raise RuntimeError("Variable updated more than once!")

    @visit.on_type(ExpandDims)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        op_element_type = self.computation.lookup_cpp_op(x)
        axis_set = set()
        axis_set.add(op.dim)
        self.computation.register_cpp_op(op, PyngBroadcast(op_element_type,
                                                           Shape(list(op.axes.lengths)),
                                                           AxisSet(axis_set)))

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
        ngraph_conv = PyngConvolution(
            self.computation.lookup_cpp_op(inputs),
            self.computation.lookup_cpp_op(filters),
            Strides([op.conv_params['str_h'], op.conv_params['str_w']]),
            Strides([op.conv_params['dil_h'], op.conv_params['dil_w']]),
            CoordinateDiff([op.conv_params['pad_h'], op.conv_params['pad_w']]),
            CoordinateDiff([op.conv_params['pad_h'], op.conv_params['pad_w']]),
            Strides([1, 1]))
        ngraph_conv.name = op.name.replace('/', '_') + "_Convolution"
        self.computation.register_cpp_op(op, ngraph_conv, set_name=False)

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
        ngraph_bprop_conv = PyngConvolutionBackpropData(
            Shape(list(data.axes.lengths)),
            self.computation.lookup_cpp_op(filters),
            self.computation.lookup_cpp_op(delta),
            Strides([conv_params['str_h'], conv_params['str_w']]),
            Strides([conv_params['dil_h'], conv_params['dil_w']]),
            CoordinateDiff([conv_params['pad_h'], conv_params['pad_w']]),
            CoordinateDiff([conv_params['pad_h'], conv_params['pad_w']]),
            Strides([1, 1]))
        ngraph_bprop_conv.name = op.name.replace('/', '_') + "_ConvolutionBackpropData"
        self.computation.register_cpp_op(op, ngraph_bprop_conv, set_name=False)

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
        ngraph_update_conv = PyngConvolutionBackpropFilters(
            self.computation.lookup_cpp_op(data),
            Shape(list(filters.axes.lengths)),
            self.computation.lookup_cpp_op(delta),
            Strides([conv_params['str_h'], conv_params['str_w']]),
            Strides([conv_params['dil_h'], conv_params['dil_w']]),
            CoordinateDiff([conv_params['pad_h'], conv_params['pad_w']]),
            CoordinateDiff([conv_params['pad_h'], conv_params['pad_w']]),
            Strides([1, 1]))
        ngraph_update_conv.name = op.name.replace('/', '_') + "_ConvolutionBackpropFilters"
        self.computation.register_cpp_op(op, ngraph_update_conv, set_name=False)

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
            ngraph_pool = PyngMaxPool(self.computation.lookup_cpp_op(inputs),
                                      Shape([op.pool_params['R'],
                                             op.pool_params['S']]),
                                      Strides([op.pool_params['str_h'],
                                               op.pool_params['str_w']]),
                                      Shape([op.pool_params['pad_h'],
                                             op.pool_params['pad_w']]),
                                      Shape([op.pool_params['pad_h'],
                                             op.pool_params['pad_w']]))
            self.computation.register_cpp_op(op, ngraph_pool, set_name=False)
        elif 'avg' == op.pool_params['op']:
            ngraph_pool = PyngAvgPool(self.computation.lookup_cpp_op(inputs),
                                      Shape([op.pool_params['R'],
                                             op.pool_params['S']]),
                                      Strides([op.pool_params['str_h'],
                                               op.pool_params['str_w']]),
                                      Shape([op.pool_params['pad_h'],
                                             op.pool_params['pad_w']]),
                                      Shape([op.pool_params['pad_h'],
                                             op.pool_params['pad_w']]),
                                      True)
            """
            print(list(op.axes.lengths))
            print(ngraph_pool.get_output_shape(0))
            """

            self.computation.register_cpp_op(op, ngraph_pool)
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
            ngraph_fprop = self.computation.lookup_cpp_op(op.fprop)
            ngraph_pool = PyngMaxPoolBackprop(self.computation.lookup_cpp_op(inputs),
                                              self.computation.lookup_cpp_op(delta),
                                              Shape([op.fprop.pool_params['R'],
                                                     op.fprop.pool_params['S']]),
                                              Strides([op.fprop.pool_params['str_h'],
                                                       op.fprop.pool_params['str_w']]),
                                              Shape([op.fprop.pool_params['pad_h'],
                                                     op.fprop.pool_params['pad_w']]),
                                              Shape([op.fprop.pool_params['pad_h'],
                                                     op.fprop.pool_params['pad_w']]),
                                              ngraph_fprop)

            self.computation.register_cpp_op(op, ngraph_pool)
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
            inputs = op.inputs
            ngraph_pool = PyngAvgPoolBackprop(Shape(list(inputs.axes.lengths)),
                                              self.computation.lookup_cpp_op(delta),
                                              Shape([op.fprop.pool_params['R'],
                                                     op.fprop.pool_params['S']]),
                                              Strides([op.fprop.pool_params['str_h'],
                                                       op.fprop.pool_params['str_w']]),
                                              Shape([op.fprop.pool_params['pad_h'],
                                                     op.fprop.pool_params['pad_w']]),
                                              Shape([op.fprop.pool_params['pad_h'],
                                                     op.fprop.pool_params['pad_w']]),
                                              True)
            self.computation.register_cpp_op(op, ngraph_pool, set_name=False)
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
        ngraph_sliced = PyngSlice(op_element_type, Coordinate(lowers),
                                  Coordinate(uppers), Strides(strides))
        if axes_to_remove:
            ngraph_sliced = PyngReshape(ngraph_sliced,
                                        AxisVector(list(range(0, len(x.axes)))),
                                        Shape(list(op.axes.lengths)))

        self.computation.register_cpp_op(op, ngraph_sliced)

    @visit.on_type(SqrtOp)
    def visit(self, op, x):
        self.unary_op(op, x)

    @visit.on_type(SquareOp)
    def visit(self, op, x):
        self.unary_op(op, x)

    @visit.on_type(Flatten)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_flatten = PyngReshape(self.computation.lookup_cpp_op(x),
                                     AxisVector(list(range(0, len(x.axes)))),
                                     Shape(list(op.axes.lengths)))
        self.computation.register_cpp_op(op, ngraph_flatten)

    @visit.on_type(Unflatten)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        self.computation.set_op_rank(op)
        ngraph_unflatten = PyngReshape(self.computation.lookup_cpp_op(x),
                                       AxisVector(list(range(0, len(x.axes)))),
                                       Shape(list(op.axes.lengths)))
        self.computation.register_cpp_op(op, ngraph_unflatten)

    @visit.on_type(ContiguousOp)
    def visit(self, op, x):
        self.computation.set_op_rank(op)
        ngraph_x = self.computation.lookup_cpp_op(x)
        self.computation.register_cpp_op(op, ngraph_x, set_name=False)

    """
    BatchNorm(double eps,
              std::shared_ptr<Node> gamma,
              std::shared_ptr<Node> beta,
              std::shared_ptr<Node> input);
    """
    @visit.on_type(BatchnormCommonOp)
    def visit(self, op, inputs, gamma, beta):
        self.computation.set_op_rank(op)
        ngraph_gamma = self.computation.lookup_cpp_op(gamma)
        ngraph_beta = self.computation.lookup_cpp_op(beta)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_bnc = PyngBatchNorm(op.epsilon, ngraph_gamma, ngraph_beta, ngraph_inputs)
        self.computation.register_cpp_op(op, ngraph_bnc)

    @visit.on_type(BatchnormOutputOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_output = PyngGetOutputElement(ngraph_inputs, 0)
        self.computation.register_cpp_op(op, ngraph_output)

    @visit.on_type(BatchnormMeanOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_mean = PyngGetOutputElement(ngraph_inputs, 1)
        self.computation.register_cpp_op(op, ngraph_mean)

    @visit.on_type(BatchnormVarOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_var = PyngGetOutputElement(ngraph_inputs, 2)
        self.computation.register_cpp_op(op, ngraph_var)

    """
    BatchNormBackprop(double eps,
                      std::shared_ptr<Node> gamma,
                      std::shared_ptr<Node> beta,
                      std::shared_ptr<Node> input,
                      std::shared_ptr<Node> mean,
                      std::shared_ptr<Node> variance,
                      std::shared_ptr<Node> delta);
    """
    @visit.on_type(BatchnormBpropCommonOp)
    def visit(self, op, inputs, gamma, beta, mean, variance, delta):
        self.computation.set_op_rank(op)
        ngraph_gamma = self.computation.lookup_cpp_op(gamma)
        ngraph_beta = self.computation.lookup_cpp_op(beta)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_mean = self.computation.lookup_cpp_op(mean)
        ngraph_var = self.computation.lookup_cpp_op(variance)
        ngraph_delta = self.computation.lookup_cpp_op(delta)
        ngraph_output = PyngBatchNormBackprop(op.epsilon,
                                              ngraph_gamma,
                                              ngraph_beta,
                                              ngraph_inputs,
                                              ngraph_mean,
                                              ngraph_var,
                                              ngraph_delta)
        self.computation.register_cpp_op(op, ngraph_output)

    @visit.on_type(BatchnormBpropDataOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_data = PyngGetOutputElement(ngraph_inputs, 0)
        self.computation.register_cpp_op(op, ngraph_data)

    @visit.on_type(BatchnormBpropGammaOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_data = PyngGetOutputElement(ngraph_inputs, 1)
        self.computation.register_cpp_op(op, ngraph_data)

    @visit.on_type(BatchnormBpropBetaOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_data = PyngGetOutputElement(ngraph_inputs, 2)
        self.computation.register_cpp_op(op, ngraph_data)

    @visit.on_type(ReluOp)
    def visit(self, op, inputs):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_relu = PyngRelu(ngraph_inputs)
        self.computation.register_cpp_op(op, ngraph_relu)

    """
    ReluBackprop(std::shared_ptr<ngraph::Node> arg, std::shared_ptr<ngraph::Node> delta);
    """
    @visit.on_type(ReluBpropOp)
    def visit(self, op, inputs, delta):
        self.computation.set_op_rank(op)
        ngraph_inputs = self.computation.lookup_cpp_op(inputs)
        ngraph_delta = self.computation.lookup_cpp_op(delta)
        ngraph_relu_bprop = PyngReluBackprop(ngraph_inputs, ngraph_delta)
        self.computation.register_cpp_op(op, ngraph_relu_bprop)
