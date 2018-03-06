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
from setuptools import setup, Extension, find_packages
import sys
import sysconfig
import os


"""
List requirements here as loosely as possible but include known limitations.

For example if we know that cffi <1.0 does not work, we list the min version here,
but for other package where no known limitations exist we do not impose a restriction.

This impacts external users who install ngraph via pip, and may install ngraph inside
an environment where an existing version of these required packages exists and should
not be upgraded/downgraded by our install unless absolutely necessary.
"""
requirements = [
    "numpy",
    "h5py",
    "appdirs",
    "six",
    "requests",
    "frozendict",
    "cached-property",
    "orderedset",
    "tqdm",
    "enum34",
    "future",
    "configargparse",
    "cachetools",
    "decorator",
    "monotonic",
    "pillow",
    "jupyter",
    "nbconvert",
    "nbformat",
    "setuptools",
    "cffi>=1.0",
    "parsel",
]


setup(
    name="neon",
    version="3.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='http://www.intelnervana.com',
    license='License :: OSI Approved :: Apache Software License',
    ext_modules=ext_modules,
    package_data={'neon': ['logging.json']},
)
