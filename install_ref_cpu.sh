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

#REF_VERSIONS_URL="https://github.com/NervanaSystems/ngraph-neon/raw/master/REF_VERSIONS"
#REF_VERSIONS=$(curl -Ls $REF_VERSIONS_URL)
#echo $REF_VERSIONS
REF_VERSIONS=$(cat $SCRIPT_DIR/REF_VERSIONS)
#echo $REF_VERSIONS
NGRAPH_VERSION=$(echo $REF_VERSIONS | awk '{print $2}')
AEON_VERSION=$(echo $REF_VERSIONS | awk '{print $4}')
NEON_VERSION=$(echo  $REF_VERSIONS | awk '{print $6}')

#echo $NGRAPH_VERSION
#echo $AEON_VERSION
#echo $NEON_VERSION

${SCRIPT_DIR}/contrib/neon/install_cpu.sh $NGRAPH_VERSION $AEON_VERSION $NEON_VERSION

