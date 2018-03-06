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

# commonly used modules.  Should these still be imported in neon frontend?
from neon import make_axes
from neon.frontends.neon.axis import ax
from neon.frontends.neon.activation import Rectlin, Rectlinclip, Identity, Explin, Normalizer, Softmax, Tanh, \
    Logistic
from neon.frontends.neon.argparser import NeonArgparser
from neon.frontends.neon.arrayiterator import *
from neon.frontends.neon.callbacks import *
# from neon.frontends.neon.callbacks2 import *
from neon.frontends.neon.layer import *
from neon.frontends.neon.model import *
from neon.frontends.neon.optimizer import *
from neon.frontends.neon.initializer import *
from neon.frontends.neon.data import *
from neon.frontends.neon.saver import *
from neon.frontends.neon.saverfile import *
