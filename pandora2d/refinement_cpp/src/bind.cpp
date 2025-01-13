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
This module contains functions associated to the binding pybind of cpp dichotomy computation.
*/

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "dichotomy.hpp"

using namespace pybind11::literals;

PYBIND11_MODULE(refinement_bind, m) {
  m.def("compute_dichotomy", &compute_dichotomy, "cost_volume"_a, "disparity_map_col"_a,
        "disparity_map_row"_a, "score_map"_a, "criteria_map"_a, "cv_size"_a, "subpixel"_a,
        "nb_iterations"_a, "filter"_a, "method_matching_cost"_a,
        R"mydelimiter(
            Dichotomy calculation

            :param cost_volume: cost volume data
            :type cost_volume: NDArray[np.floating]
            :param disparity_map_col: column disparity map data
            :type disparity_map_col: NDArray[np.floating]
            :param disparity_map_row: row disparity map data
            :type disparity_map_row: NDArray[np.floating]
            :param score_map: score map data
            :type score_map: NDArray[np.floating]
            :param criteria_map: criteria map data
            :type criteria_map: NDArray[np.floating]
            :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
            :type cv_size: Cost_volume_size
            :param subpixel: sub-sampling of cost_volume
            :type subpixel: int
            :param nb_iterations: number of iterations of the dichotomy
            :type nb_iterations: int
            :param filter: interpolation filter
            :type filter: abstractfilter::AbstractFilter
            :param method_matching_cost: max or min
            :type method_matching_cost: str
            )mydelimiter");

  pybind11::class_<Cost_volume_size>(m, "Cost_volume_size")
      .def(pybind11::init<>())
      .def(pybind11::init<Eigen::VectorXd&>())
      .def(pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int>())
      .def_property_readonly("size", &Cost_volume_size::size, R"mydelimiter(
            Returns the cost_volume size nb_row * nb_col * nb_disp_row * nb_disp_col.
            )mydelimiter")
      .def_property_readonly("nb_disps", &Cost_volume_size::nb_disps, R"mydelimiter(
            Returns the disparity number : nb_disp_row * nb_disp_col.
            )mydelimiter");
}