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
#include "ngraph/ops/op.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_MODULE(Op, mod) {

    py::module::import("nwrapper.ngraph.Node");

    py::class_<RequiresTensorViewArgs, std::shared_ptr<RequiresTensorViewArgs>, Node> requiresTensorViewArgs(mod, "RequiresTensorViewArgs");
    py::class_<UnaryElementwise, std::shared_ptr<UnaryElementwise>,
        RequiresTensorViewArgs> unaryElementwise(mod, "UnaryElementwise");
    py::class_<UnaryElementwiseArithmetic, std::shared_ptr<UnaryElementwiseArithmetic>, 
        UnaryElementwise> unaryElementwiseArithmetic(mod, "UnaryElementwiseArithmetic"); 
    py::class_<BinaryElementwise, std::shared_ptr<BinaryElementwise>,
        RequiresTensorViewArgs> binaryElementwise(mod, "BinaryElementwise");
    py::class_<BinaryElementwiseComparison, std::shared_ptr<BinaryElementwiseComparison>,
        BinaryElementwise> binaryElementwiseComparison(mod, "BinaryElementwiseComparison");
    py::class_<BinaryElementwiseArithmetic, std::shared_ptr<BinaryElementwiseArithmetic>,
        BinaryElementwise> binaryElementwiseArithmetic(mod, "BinaryElementwiseArithmetic");    

}

}}  // ngraph