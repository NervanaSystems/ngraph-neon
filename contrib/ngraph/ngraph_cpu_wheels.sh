#!/bin/bash
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
set -e

if [ -d build ]; then
    rm -rf build
fi

lcores=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)

mkdir build && cd build && cmake .. && make -j$lcores
virtualenv .venv2 && . .venv2/bin/activate && pip install wheel setuptools \
    python setup.py bdist_wheel && deactivate && mv dist/*.whl ..
python3 -m venv .venv3 && . .venv3/bin/activate && pip install wheel setuptools \
    python setup.py bdist_wheel && deactivate && mv dist/*.whl ..

