// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/ops/sum.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_MODULE(Sum, mod) {

    py::module::import("nwrapper.ngraph.ops.Op");

    using AxisSet = std::set<size_t>;

    py::class_<Sum, std::shared_ptr<Sum>, RequiresTensorViewArgs> sum(mod, "Sum");
    sum.def(py::init<const std::shared_ptr<ngraph::Node>&,
                     const AxisSet& >());
}

}}  // ngraph