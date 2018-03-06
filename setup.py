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
    "numpy==1.13.1",
    "h5py==2.6.0",
    "appdirs==1.4.0",
    "six==1.10.0",
    "tensorflow==0.12.1",
    "scipy==0.18.1",
    "protobuf==3.2.0",
    "requests==2.13.0",
    "frozendict==1.2",
    "cached-property==1.3.0",
    "orderedset==2.0",
    "tqdm==4.11.2",
    "enum34==1.1.6",
    "future==0.16.0",
    "configargparse==0.11.0",
    "cachetools==2.0.0",
    "decorator==4.0.11",
    "monotonic==1.3",
    "jupyter==1.0.0",
    "nbconvert==5.1.1",
    "nbformat==4.3.0",
    "setuptools",
    "cffi>=1.0",
    "parsel==1.2.0",
    "pillow==4.2.0",
]


setup(
    name="neon",
    version="3.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='http://www.intelnervana.com',
    license='License :: OSI Approved :: Apache Software License',
    package_data={'neon': ['logging.json']},
)
