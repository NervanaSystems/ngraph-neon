# ----------------------------------------------------------------------------
# Copyright 2018 Nervana Systems Inc.
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
"""Provide a layer of abstraction for the ngraph++ runtime environment."""
import numpy as np

import nwrapper.ngraph.Util as Util
import nwrapper.ngraph.types.TraitedType as TraitedType
import nwrapper.ngraph.types.TensorViewType as TensorViewType
import nwrapper.ngraph.Node as Node
import nwrapper.ngraph.Function as Function
import nwrapper.ngraph.runtime.Manager as Manager
from nwrapper.ngraph.ops.Parameter import Parameter

from ngraph.types import py_numeric_type


def runtime(manager_name: str='NGVM') -> 'Runtime':
    """Helper factory to create a Runtime object.

    Use signature to parametrize runtime as needed."""
    return Runtime(manager_name)


class Runtime:
    """Represents the ngraph++ runtime environment."""

    def __init__(self, manager_name: str) -> None:
        self.manager_name = manager_name
        self.manager = Manager.Manager.get(manager_name)
        self.backend = self.manager.allocate_backend()

    def __repr__(self):
        return '<Runtime: Manager=\'{}\'>'.format(self.manager_name)

    def computation(self, node: Node, *inputs: py_numeric_type) -> 'Computation':
        return Computation(self, node, *inputs)


class Computation:
    """ngraph callable computation object"""

    def __init__(self, runtime: Runtime, node: Node, *parameters: Parameter) -> None:
        self.runtime = runtime
        self.node = node
        self.parameters = parameters
        self.tensor_views = []
        for parameter in parameters:
            shape = parameter.get_shape()
            element_type = TraitedType.TraitedTypeF.element_type()  # @TODO: get from input
            self.tensor_views.append(runtime.backend.make_primary_tensor_view(element_type, shape))

    def __repr__(self) -> str:
        params_string = ', '.join([param.name for param in self.parameters])
        return '<Computation: {}({})>'.format(self.node.name, params_string)

    def __call__(self, *input_values: py_numeric_type) -> py_numeric_type:
        """This is a quick and dirty implementation of the logic needed to calculate and return
        a value of the computation."""
        for tensor_view, value in zip(self.tensor_views, input_values):
            if not type(value) == np.ndarray:
                value = np.array(value)
            tensor_view.write(Util.numpy_to_c(value), 0, 16)

        result_element_type = TraitedType.TraitedTypeF.element_type()  # @TODO: get from self.node
        result_shape = [2, 2]  # @TODO: get from self.node

        result = self.runtime.backend.make_primary_tensor_view(result_element_type, result_shape)
        result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
        result.write(Util.numpy_to_c(result_arr), 0, 16)

        value_type = TensorViewType.TensorViewType(result_element_type, result_shape)
        function = Function.Function(self.node, value_type, self.parameters, 'test')
        external = self.runtime.manager.compile(function)

        cf = self.runtime.backend.make_call_frame(external)
        cf.call(self.tensor_views, [result])

        result.read(Util.numpy_to_c(result_arr), 0, 16)
        result_arr = result_arr.reshape(result_shape)
        return result_arr
