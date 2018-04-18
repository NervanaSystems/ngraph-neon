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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEON_ROOT=$SCRIPT_DIR/../..

if [ -z "$1" ]; then
    echo "Building nGraph master"
    NGRAPH_VERSION="master"
else
    echo "Building nGraph version $1"
    NGRAPH_VERSION="$1"
fi

if [ -z "$2" ]; then
    echo "Building neon master"
    NEON_VERSION="master"
else
    echo "Building neon version $2"
    NEON_VERSION="$2"
fi

lcores=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)

if [ -x "$(command -v lsb_release)" ]; then
    DISTRIB_ID=$(lsb_release -is)
fi

if [ -n "$DISTRIB_ID" ]; then
    if [ "$DISTRIB_ID" == "Ubuntu" ]; then
        ubuntu_ver=$(lsb_release -rs)
        if [ "$ubuntu_ver" == "16.04" ]; then
            cmake -DNGRAPH_VERSION=$NGRAPH_VERSION -DNEON_VERSION=$NEON_VERSION -DNEON_ROOT=$NEON_ROOT -DNGRAPH_USE_PREBUILT_LLVM=TRUE $SCRIPT_DIR && make -j$lcores
        else
            cmake -DNGRAPH_VERSION=$NGRAPH_VERSION -DNEON_VERSION=$NEON_VERSION -DNEON_ROOT=$NEON_ROOT $SCRIPT_DIR && make -j$lcores
        fi
    else # Linux but not Ubuntu
        cmake -DNGRAPH_VERSION=$NGRAPH_VERSION -DNEON_VERSION=$NEON_VERSION -DNEON_ROOT=$NEON_ROOT $SCRIPT_DIR && make -j$lcores
    fi
else # Not Linux
    cmake -DNGRAPH_VERSION=$NGRAPH_VERSION -DNEON_VERSION=$NEON_VERSION -DNEON_ROOT=$NEON_ROOT $SCRIPT_DIR && make -j$lcores
fi

virtualenv -p python2.7 .venv2 && . .venv2/bin/activate && pip install -U pip wheel setuptools && python setup.py bdist_wheel && deactivate && mv dist/*.whl .
python3 -m venv .venv3 && . .venv3/bin/activate && pip install -U pip wheel setuptools && python setup.py bdist_wheel && deactivate && mv dist/*.whl .

