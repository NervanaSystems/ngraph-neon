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
import logging

import numpy as np

from pyngraph import Function, Node, util
from pyngraph.runtime import Manager
from pyngraph.op import Parameter

from ngraph_api.utils.types import py_numeric_type, get_dtype

log = logging.getLogger(__file__)


def runtime(manager_name: str='INTERPRETER') -> 'Runtime':
    """Helper factory to create a Runtime object.

    Use signature to parametrize runtime as needed."""
    return Runtime(manager_name)


class Runtime:
    """Represents the ngraph++ runtime environment."""

    def __init__(self, manager_name: str) -> None:
        self.manager_name = manager_name
        self.manager = Manager.get(manager_name)
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
            element_type = parameter.get_element_type()
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
            Computation._write_ndarray_to_tensor_view(value, tensor_view)

        result_element_type = self.node.get_element_type()
        result_shape = self.node.get_shape()
        result_dtype = get_dtype(result_element_type)

        result_view = self.runtime.backend.make_primary_tensor_view(result_element_type, result_shape)
        result_arr = np.empty(result_shape, dtype=result_dtype)

        function = Function(self.node, self.parameters, 'ngraph API computation')
        external = self.runtime.manager.compile(function)
        call_frame = self.runtime.backend.make_call_frame(external)
        call_frame.call(self.tensor_views, [result_view])

        Computation._read_tensor_view_to_ndarray(result_view, result_arr)
        result_arr = result_arr.reshape(result_shape)
        return result_arr

    @staticmethod
    def _get_buffer_size(element_type, element_count):
        return int((element_type.bitwidth / 8) * element_count)

    @staticmethod
    def _write_ndarray_to_tensor_view(value: np.ndarray, tensor_view):
        tensor_view_dtype = get_dtype(tensor_view.element_type)
        if value.dtype != tensor_view_dtype:
            log.warning('Attempting to write a %s value to a %s tensor. Will attempt type conversion.',
                        value.dtype, tensor_view.element_type)
            value = value.astype(tensor_view_dtype, casting='safe')

        buffer_size = Computation._get_buffer_size(tensor_view.element_type, tensor_view.element_count)
        tensor_view.write(util.numpy_to_c(value), 0, buffer_size)

    @staticmethod
    def _read_tensor_view_to_ndarray(tensor_view, output: np.ndarray):
        buffer_size = Computation._get_buffer_size(tensor_view.element_type, tensor_view.element_count)
        tensor_view.read(util.numpy_to_c(output), 0, buffer_size)
