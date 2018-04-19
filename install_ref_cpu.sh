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

REF_VERSIONS="https://github.com/NervanaSystems/ngraph-neon/raw/master/REF_VERSIONS"

NGRAPH_VERSION=$(curl -Ls ${REF_VERSIONS} | sed -n '1p')
AEON_VERSION=$(curl -Ls ${REF_VERSIONS} | sed -n '2p')
NEON_VERSION=$(curl -Ls ${REF_VERSIONS} | sed -n '3p')

${SCRIPT_DIR}/contrib/neon/install_cpu.sh $NGRAPH_VERSION $AEON_VERSION $NEON_VERSION

