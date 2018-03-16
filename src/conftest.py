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
import pytest
import neon.transformers as ngt


def pytest_addoption(parser):
    parser.addoption("--batch_size", type=int, default=8,
                     help="Batch size for tests using input_tensor fixture.")
    parser.addoption("--transformer", default="ngcpu", choices=ngt.transformer_choices(),
                     help="Select from available transformers")
    parser.addoption("--serialization_integration_test", action="store_true",
                     help="Force all unit tests to serialize and deserialize the graph before \
                     transformer compilation.")


def pytest_xdist_node_collection_finished(node, ids):
    ids.sort()


def pytest_generate_tests(metafunc):
    pass


@pytest.fixture(scope="module", autouse=True)
def transformer_factory(request):
    def set_and_get_factory(transformer_name):
        factory = ngt.make_transformer_factory(transformer_name)
        ngt.set_transformer_factory(factory)
        return factory

    name = request.config.getoption("--transformer")

    yield set_and_get_factory(name)

    # Reset transformer factory to default
    ngt.set_transformer_factory(ngt.make_transformer_factory("ngcpu"))


def pass_method(*args, **kwargs):
    pass


def pytest_configure(config):

    # when marking argon_disabled for a whole test, but flex_disabled only on one
    # parametrized version of that test, the argon marking disappeared
    config.ngargon_skip = pytest.mark.skipif(config.getvalue("transformer") == "ngargon")
    config.ngcpu_skip = pytest.mark.skipif(config.getvalue("transformer") == "ngcpu")
    config.nggpu_skip = pytest.mark.skipif(config.getvalue("transformer") == "nggpu")
