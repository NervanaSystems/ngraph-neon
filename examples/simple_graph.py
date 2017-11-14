# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# ----------------------------------------------------------------------------
from contextlib import closing
import ngraph as ng


def simple_graph():
    a = ng.constant(0.)
    b = ng.constant(1.)

    c = ng.add(a, b)

    with closing(ng.executor()) as ex:
        c_val = ex.execute(c)
        print(c_val)


if __name__ == "__main__":
    simple_graph()
