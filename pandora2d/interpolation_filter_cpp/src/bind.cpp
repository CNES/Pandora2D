/* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA2D
 *
 *     https://github.com/CNES/Pandora2D
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This module contains functions associated to the binding pybind of cpp filter methods.
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "bicubic.hpp"

namespace py = pybind11;

PYBIND11_MODULE(interpolation_filter_bind, m){

    py::class_<abstractfilter::AbstractFilter>(m, "AbstractFilter")
    .def("get_coeffs", &abstractfilter::AbstractFilter::get_coeffs)
    .def("apply", &abstractfilter::AbstractFilter::apply)
    .def("interpolate", &abstractfilter::AbstractFilter::interpolate)
    .def("get_margins", &abstractfilter::AbstractFilter::get_margins);

    py::class_<Bicubic,abstractfilter::AbstractFilter>(m, "Bicubic")
    .def(py::init<>())
    .def("get_coeffs", &Bicubic::get_coeffs);

    py::class_<Margins>(m, "Margins")
    .def_readwrite("up", &Margins::up)
    .def_readwrite("down", &Margins::down)
    .def_readwrite("left", &Margins::left)
    .def_readwrite("right", &Margins::right);



}


// Compilation command
// g++ -I eigen-3.4.0 -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) 
// pandora2d/interpolation_filter_cpp/src/bicubic.cpp 
// pandora2d/interpolation_filter_cpp/src/interpolation_filter.cpp 
// pandora2d/interpolation_filter_cpp/src/bind.cpp 
// -o pandora2d/interpolation_filter_cpp/src/filter_bindings_cpp$(python3-config --extension-suffix)