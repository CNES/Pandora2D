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
This module contains functions associated to the binding pybind of cpp dichotomy computation.
*/

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "dichotomy.hpp"
#include "pandora2d_type.hpp"

using namespace pybind11::literals;

PYBIND11_MODULE(refinement_bind, m) {
  m.def("compute_dichotomy_float", &compute_dichotomy<float, P2d::Vectorf>, "cost_volume"_a,
        "disparity_map_col"_a, "disparity_map_row"_a, "score_map"_a, "criteria_map"_a, "subpixel"_a,
        "nb_iterations"_a, "filter"_a, "method_matching_cost"_a,
        R"mydelimiter(
            Dichotomy calculation with float data

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
            :param subpixel: sub-sampling of cost_volume
            :type subpixel: int
            :param nb_iterations: number of iterations of the dichotomy
            :type nb_iterations: int
            :param filter: interpolation filter
            :type filter: abstractfilter::AbstractFilter
            :param method_matching_cost: max or min
            :type method_matching_cost: str
            )mydelimiter");

  m.def("compute_dichotomy_double", &compute_dichotomy<double, P2d::VectorD>, "cost_volume"_a,
        "disparity_map_col"_a, "disparity_map_row"_a, "score_map"_a, "criteria_map"_a, "subpixel"_a,
        "nb_iterations"_a, "filter"_a, "method_matching_cost"_a,
        R"mydelimiter(
            Dichotomy calculation with double data

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
            :param subpixel: sub-sampling of cost_volume
            :type subpixel: int
            :param nb_iterations: number of iterations of the dichotomy
            :type nb_iterations: int
            :param filter: interpolation filter
            :type filter: abstractfilter::AbstractFilter
            :param method_matching_cost: max or min
            :type method_matching_cost: str
            )mydelimiter");
}