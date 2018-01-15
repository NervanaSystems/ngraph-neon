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
"""Functions related to converting between Python and numpy types and ngraph types"""
from typing import Union

import numpy as np

import nwrapper.ngraph.types.TraitedType as TraitedType

py_numeric_type = Union[int, float, np.ndarray]


def get_element_type(dtype: py_numeric_type):
    """Return an ngraph element type for a Python type or numpy.dtype."""
    # @TODO: Map all types
    element_type = TraitedType.TraitedTypeF.element_type()
    return element_type
