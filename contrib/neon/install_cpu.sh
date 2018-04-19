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

if [ -z "$1" ]; then
    echo "Building nGraph master"
    NGRAPH_VERSION="master"
else
    echo "Building nGraph version $1"
    NGRAPH_VERSION="$1"
fi

if [ -z "$2" ]; then
    echo "Building aeon master"
    AEON_VERSION="master"
else
    echo "Building aeon version $2"
    AEON_VERSION="$2"
fi

if [ -z "$3" ]; then
    echo "Building neon from current repo"
    ${SCRIPT_DIR}/neon_cpu_wheels.sh $NGRAPH_VERSION
else
    echo "Building neon version $3"
    NEON_VERSION="$3"
    ${SCRIPT_DIR}/neon_cpu_wheels.sh $NGRAPH_VERSION $NEON_VERSION
fi


PY2_WHEEL="$(find . -name ngraph_neon*cp2*.whl)"
PY3_WHEEL="$(find . -name ngraph_neon*cp3*.whl)"

git clone -n https://github.com/NervanaSystems/aeon.git
cd aeon && git checkout ${AEON_VERSION} && cd ..

. .venv2/bin/activate && pip install ${PY2_WHEEL} && mkdir -p aeon/build2 && pushd aeon/build2 && cmake .. && pip install -U . && deactivate && popd && echo "neon for python2 installed in virtualenv .venv2"

. .venv3/bin/activate && pip install ${PY3_WHEEL} && mkdir -p aeon/build3 && pushd aeon/build3 && cmake .. && pip install -U . && deactivate && popd && echo "neon for python3 installed in virtualenv .venv3"
