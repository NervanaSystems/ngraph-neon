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

#include <iterator>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/operators.h>
//#include <string>
#include "ngraph/node.hpp"
#include "ngraph/ops/parameter.hpp"
#include "pyngraph/ops/parameter.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Parameter(py::module m){

    //py::module::import("wrapper.ngraph.types.TraitedType");
    //py::module::import("wrapper.ngraph.Node");
    py::class_<ngraph::op::Parameter, std::shared_ptr<ngraph::op::Parameter>, ngraph::Node> parameter(m, "Parameter");

    parameter.def("__repr__", [](const ngraph::Node &self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();

        std::stringstream shape_string_stream;
        std::copy(self.get_shape().begin(), self.get_shape().end(), std::ostream_iterator<int>(shape_string_stream, ", "));
        std::string shape = shape_string_stream.str();
        std::string type = self.get_element_type().c_type_string();

        return "<" + class_name + ": '" + self.get_name() + "' (" + shape + type + ")>";
    });

    parameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape& >());
    parameter.def_property_readonly("description", &ngraph::op::Parameter::description);
}

