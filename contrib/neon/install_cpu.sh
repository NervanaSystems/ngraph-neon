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

${SCRIPT_DIR}/neon_cpu_wheels.sh

PY2_WHEEL="$(find . -name ngraph_neon*cp2*.whl)"
PY3_WHEEL="$(find . -name ngraph_neon*cp3*.whl)"

read AEON_VERSION < "$NEON_ROOT/aeon.version"

git clone -b ${AEON_VERSION} https://github.com/NervanaSystems/aeon.git

. .venv2/bin/activate && pip install ${PY2_WHEEL} && mkdir -p aeon/build2 && pushd aeon/build2 && cmake .. && pip install -U . && deactivate && popd && echo "neon for python2 installed in virtualenv .venv2"

. .venv3/bin/activate && pip install ${PY3_WHEEL} && mkdir -p aeon/build3 && pushd aeon/build3 && cmake .. && pip install -U . && deactivate && popd && echo "neon for python3 installed in virtualenv .venv3"
