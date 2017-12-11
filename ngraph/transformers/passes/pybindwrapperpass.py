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
from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, Add, Multiply, BroadcastOp, TensorValueOp
import nwrapper.ngraph.types.TraitedType as TraitedType
import nwrapper.ngraph.ops.Parameter as Parameter
import nwrapper.ngraph.runtime.ParameterizedTensorView as ParameterizedTensorView
import nwrapper.ngraph.Util as Util
import nwrapper.ngraph.runtime.Utils as Utils
import nwrapper.ngraph.ops.ParameterizedConstant as ParameterizedConstant
import numpy as np
import nwrapper.ngraph.ops.Broadcast as Broadcast


class PybindWrapperGenerator(PeepholeGraphPass):
    """
    Graph pass to generate the PybindWrapper's by visiting all the Op's
    needed to compute the results.

    Arguments
        transformer (obj:`Transformer`): The associated transformer.
    """

    def __init__(self, tranformer, **kwargs):
        super(PybindWrapperGenerator, self).__init__(**kwargs)
        self.transformer = tranformer

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        pass

    @visit.on_type(Add)
    def visit(self, op, x, y):
        ngraph_cpp_add_op = self.transformer.ngraph_cpp_op_prameter[x.tensor] \
            + self.transformer.ngraph_cpp_op_prameter[y.tensor]

        self.transformer.ngraph_cpp_op_prameter[op.tensor] = ngraph_cpp_add_op

    @visit.on_type(Multiply)
    def visit(self, op, x, y):
        ngraph_cpp_mul_op = self.transformer.ngraph_cpp_op_prameter[x.tensor] \
            * self.transformer.ngraph_cpp_op_prameter[y.tensor]

        self.transformer.ngraph_cpp_op_prameter[op.tensor] = ngraph_cpp_mul_op

    @visit.on_type(BroadcastOp)
    def visit(self, op, input):
        element_type = TraitedType.TraitedTypeF.element_type()
        # check if the op.args already have Paramterized view type.
        if op.args[0].tensor in self.transformer.ngraph_cpp_op_prameter:
            op_element_type =  self.transformer.ngraph_cpp_op_prameter[op.args[0].tensor]
        else:
            op_element_type = Parameter.Parameter(
                element_type, list(op.args[0].axes.lengths))
        axis_set = list(range(len(op.axes.lengths)))
        self.transformer.ngraph_cpp_op_prameter[op.tensor] = \
            Broadcast.Broadcast(op_element_type, list(op.axes.lengths), set(axis_set))


    @visit.on_type(TensorValueOp)
    def visit(self, op):

        if op.tensor not in self.transformer.ngraph_cpp_op_prameter:
            if op.tensor.is_constant:
                constant_tensor = Utils.make_tensor(list(op.axes.lengths))
                item_size = op.tensor.dtype.itemsize
                element_size = (op.tensor.const.size) * item_size
                constant_tensor.write(Util.numpy_to_c(
                    np.array([op.tensor.const.__abs__()], dtype=np.float32)), 0, element_size)
                constant_op = ParameterizedConstant.ParameterizedConstantF(list(op.axes.lengths), constant_tensor)
                self.transformer.ngraph_cpp_op_prameter[op.tensor] = constant_op
            else:
                element_type = TraitedType.TraitedTypeF.element_type()
                op_element_type = Parameter.Parameter(
                    element_type, list(op.axes.lengths))
                self.transformer.ngraph_cpp_op_prameter[op.tensor] = op_element_type


