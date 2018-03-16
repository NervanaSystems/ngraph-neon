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
from neon.frontend.axis import ax
from neon.frontend.activation import Rectlin, Rectlinclip, Identity, Explin, Normalizer, Softmax, Tanh, \
    Logistic
from neon.frontend.argparser import NeonArgparser
from neon.frontend.arrayiterator import *
from neon.frontend.callbacks import *
# from neon.frontend.callbacks2 import *
from neon.frontend.layer import *
from neon.frontend.model import *
from neon.frontend.optimizer import *
from neon.frontend.initializer import *
from neon.frontend.data import *
from neon.frontend.saver import *
from neon.frontend.saverfile import *
