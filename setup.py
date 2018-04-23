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
from setuptools import setup, find_packages
from subprocess import Popen, PIPE
import os, sys

p = Popen(['git', 'describe', '--tags'], cwd=os.getcwd(), stdout=PIPE)
out, err = p.communicate()
if sys.version_info.major > 2:
    out = str(out, 'utf-8')
__version__ = out.split('-')[0][1:]

with open('requirements.txt') as req:
    requirements = req.read().splitlines()

setup(
    name="ngraph-neon",
    version=__version__,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='http://www.intelnervana.com',
    license='License :: OSI Approved :: Apache Software License',
    package_data={'neon': ['logging.json']},
)
