.. _installation:

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

Installation
############

Requirements
============

Intel® Neon requires **Python 2.7** or **Python 3.4+** running on a 
Linux* or UNIX-based OS. Before installing, also ensure your system has recent 
updates of the following packages:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   libyaml-dev, pyaml, Parses YAML format inputs
   pkg-config, pkg-config, Retrieves information about installed libraries
   pandoc, pandoc, Only required for building documentation


Prerequisites  
=============

#. **Choose your build environment.** Installing within a virtual environment
   is the easiest option for most users. To prepare for a system installation,
   you may skip this step.  

   * **Python3** 
     To create and activate a Python 3 virtualenv:
     
    .. code-block:: console
   
       $ python3 -m venv .venv
       $ . .venv/bin/activate

   * **Python 2.7**
     To create and activate a Python 2 virtualenv:

    .. code-block:: console

       $ virtualenv -p python2.7 .venv
       $ . .venv/bin/activate

#. **Download the source code.**

    .. code-block:: console

       $ git clone https://github.com/NervanaSystems/ngraph-neon.git
       $ cd ngraph-neon




Installation
============
  
To build and install Intel Neon, keep the virtualenv activated and 
simply run ``make install`` from within the clone of the repo as follows:

.. code-block:: console

   $ make install



Back-end Configuration
======================

After completing the prerequisites and installation of the base Neon
Graph package, Intel® nGraph™ and the python wraooer needs to be installed.
Clone the source code from https://github.com/NervanaSystems/ngraph
Activate the virtualenv if you haven't already. And follow the instructions at
https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
Once the python wheel is built

.. code-block:: console

    $ pip install <path-to-python-wheel>/<wheel-name>.whl




Getting Started
===============

Some Jupyter* notebook walkthroughs demonstrate ways to use Intel Nervana Graph:

* ``examples/walk_through/``: Use Nervana Graph to implement logistic regression 
* ``examples/mnist/MNIST_Direct.ipynb``: Build a deep learning model directly on 
  Neon Graph

The neon framework can also be used to define and train deep learning models:

* ``examples/mnist/mnist_mlp.py``: Multilayer perceptron network on MNIST dataset.
* ``examples/cifar10/cifar10_conv.py``: Convolutional neural network on CIFAR-10.
* ``examples/cifar10/cifar10_mlp.py``: Multilayer perceptron on CIFAR-10 dataset.

Developer Guidelines
====================

Before checking in code, run the unit tests and check for style errors:

.. code-block:: console

   $ make test
   $ make style

Documentation can be generated with pandoc:

.. code-block:: console

   $ sudo apt-get install pandoc
   $ make doc

View the documentation at ``doc/build/html/index.html``.


