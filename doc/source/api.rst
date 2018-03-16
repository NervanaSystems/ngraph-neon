.. _api:

.. ---------------------------------------------------------------------------
.. Copyright 2017-2018 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Intel® Neon
*********************

This API documentation covers the public API for Intel® Neon, organized into three main modules:

- ``neon``: Contains the core ops for constructing the graph.
- ``neon.transformers``: Defines methods for executing a defined graph on hardware.
- ``neon.types``: Types in neon (for example, ``Axes``, ``Op``, etc.)

Intel Neon API 
==========================

Several ops are used to create different types of tensors:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`neon.variable` | Create a trainable variable.
	:meth:`neon.persistent_tensor` | Tensor that persists across computations.
	:meth:`neon.placeholder` | Used for input values, typically from host.
	:meth:`neon.constant` | Immutable constant that can be inlined.

Assigning the above tensors requires defining ``Axis``, which can be done using the following methods:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`neon.axes_with_order` | Return a tensor with a different axes order.
	:meth:`neon.cast_axes` | Cast the axes of a tensor to new axes.
	:meth:`neon.make_axes` | Create an Axes object.
	:meth:`neon.make_axis` | Create an Axis.

We also provide several helper function for retrieving information from tensors.

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

	:meth:`neon.batch_size` | Returns the batch size
	:meth:`neon.is_constant` | Returns true if tensor is constant
	:meth:`neon.is_constant_scalar` | Returns true if tensor is a constant scalar
	:meth:`neon.constant_value` | Returns the value of a constant tensor
	:meth:`neon.tensor_size` | Returns the total size of the tensor

To compose a computational graph, we support the following operations:

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`neon.absolute` | :math:`\operatorname{abs}(a)`
    :meth:`neon.negative` | :math:`-a`
	:meth:`neon.sign` | if :math:`x<0`, :math:`-1`; if :math:`x=0`, :math:`0`; if :math:`x>0`, :math:`1`
	:meth:`neon.add` | :math:`a+b`
	:meth:`neon.reciprocal` | :math:`1/a`
	:meth:`neon.square` | :math:`a^2`
	:meth:`neon.sqrt` | :math:`\sqrt{a}`
	:meth:`neon.cos` | :math:`\cos(a)`
	:meth:`neon.sin` | :math:`\sin(a)`
	:meth:`neon.tanh` | :math:`\tanh(a)`
	:meth:`neon.sigmoid` | :math:`1/(1+\exp(-a))`
	:meth:`neon.exp` | :math:`\exp(a)`
	:meth:`neon.log` | :math:`\log(a)`
	:meth:`neon.safelog` | :math:`\log(a)`
	:meth:`neon.one_hot` | Convert to one-hot
	:meth:`neon.variance` | Compute variance
	:meth:`neon.stack` | Stack tensors along an axis
	:meth:`neon.convolution` | Convolution operation
	:meth:`neon.pad` | Pad a tensor with zeros along each dimension
	:meth:`neon.pooling` | Pooling operation
	:meth:`neon.squared_L2` | dot x with itself


.. Note::
   Additional operations are supported that are not currently documented, and so are not included in the list above. We will continue to populate this API when the documentation is updated.

neon.transformers
===================

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`neon.transformers.allocate_transformer` | Allocate a transformer.
    :meth:`neon.transformers.make_transformer` | Generates a transformer using the factory.
    :meth:`neon.transformers.make_transformer_factory` | Creates a new factory with cpu default.
    :meth:`neon.transformers.set_transformer_factory` | Sets the Transformer factory used by make_transformer.
    :meth:`neon.transformers.transformer_choices` | Return the list of available transformers.
    :meth:`neon.transformers.Transformer` | Produce an executable version of op-graphs.

neon.types
============

.. csv-table::
    :header: "Method", "Description"
    :widths: 30, 70
    :delim: |

    :meth:`neon.types.AssignableTensorOp` | Assign a tensor. Used by `ng.placeholder`, and more.
    :meth:`neon.types.Axis` | An Axis labels a dimension of a tensor.
    :meth:`neon.types.Axes` | Axes represent multiple axis dimensions.
    :meth:`neon.types.Computation` | Computations to attach to transformers.
    :meth:`neon.types.NameableValue` | Objects that can derive name from the name scope.
    :meth:`neon.types.NameScope` | Name scope for objects.
    :meth:`neon.types.Op` | Basic class for ops.
    :meth:`neon.types.TensorOp` | Base class for ops related to Tensors.

neon
------

Graph construction.

.. py:module: neon

.. automodule:: neon
   :members:

neon.transformers
-------------------

Transformer manipulation.

.. py:module: neon.transformers

.. automodule:: neon.transformers
   :members:

neon.types
------------

.. py:module: neon.types

.. automodule:: neon.types
   :members:
