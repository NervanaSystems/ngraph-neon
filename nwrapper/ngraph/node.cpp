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
#include "ngraph/node.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/subtract.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_MODULE(Node, mod) {

    py::class_<Node, std::shared_ptr<Node>> node(mod, "Node");
 
    node.def("__repr__", [](const Node &self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        return "<" + class_name + ": '" + self.get_name() + "'>";
    });

    node.def("__add__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a + b;
               }, py::is_operator());
    node.def("__sub__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a - b;
               }, py::is_operator());
    node.def("__mul__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a * b;
               }, py::is_operator());
    node.def("__truediv__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a/b;
               }, py::is_operator());
    node.def("get_shape", &Node::get_shape);
    node.def("get_value_type", (std::shared_ptr<const ValueType> (Node::*)()) &Node::get_value_type);
    node.def("get_value_type", (const std::shared_ptr<const ValueType> (Node::*)() const) &Node::get_value_type);

    node.def_property_readonly("shape", &Node::get_shape);

    node.def_property("name", &Node::get_name, &Node::set_name);
}

}  // ngraph
