/* Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the binding pybind of cpp common tools.
*/

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "cost_volume.hpp"
#include "pandora2d_type.hpp"

using namespace pybind11::literals;

PYBIND11_MODULE(common_bind, m) {
  pybind11::class_<CostVolumeSize>(m, "CostVolumeSize")
      .def(pybind11::init<>())
      .def(pybind11::init<P2d::VectorD&>())
      .def(pybind11::init<std::vector<size_t>&>())
      .def(pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def_property_readonly("size", &CostVolumeSize::size, R"mydelimiter(
            Returns the cost_volume size nb_row * nb_col * nb_disp_row * nb_disp_col.
            )mydelimiter")
      .def_property_readonly("nb_disps", &CostVolumeSize::nb_disps, R"mydelimiter(
            Returns the disparity number : nb_disp_row * nb_disp_col.
            )mydelimiter");
}