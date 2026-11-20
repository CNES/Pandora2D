/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
  m.def("compute_dichotomy_cost_surface_float",
        &compute_dichotomy_cost_surface<float, P2d::Vectorf>, "cost_volumes"_a,
        "disparity_map_col"_a, "disparity_map_row"_a, "score_map"_a, "invalid_map"_a.noconvert(),
        "criteria_map"_a, "subpixel"_a, "nb_iterations"_a, "filter"_a,
        "type_method_matching_cost"_a,
        R"mydelimiter(
                Dichotomy refinement by resampling the cost surface (float data).

                :param cost_volumes: cost volumes data
                :type cost_volumes: NDArray[np.floating]
                :param disparity_map_col: column disparity map data
                :type disparity_map_col: NDArray[np.floating]
                :param disparity_map_row: row disparity map data
                :type disparity_map_row: NDArray[np.floating]
                :param score_map: score map data
                :type score_map: NDArray[np.floating]
                :param invalid_map: invalid map data
                :type invalid_map: NDArray[np.uint8]
                :param criteria_map: criteria map data
                :type criteria_map: NDArray[np.floating]
                :param subpixel: sub-sampling of cost_volume
                :type subpixel: int
                :param nb_iterations: number of iterations of the dichotomy
                :type nb_iterations: int
                :param filter: interpolation filter
                :type filter: abstractfilter::AbstractFilter
                :param type_method_matching_cost: max or min
                :type type_method_matching_cost: str
            )mydelimiter");

  m.def("compute_dichotomy_cost_surface_double",
        &compute_dichotomy_cost_surface<double, P2d::VectorD>, "cost_volumes"_a,
        "disparity_map_col"_a, "disparity_map_row"_a, "score_map"_a, "invalid_map"_a.noconvert(),
        "criteria_map"_a, "subpixel"_a, "nb_iterations"_a, "filter"_a,
        "type_method_matching_cost"_a,
        R"mydelimiter(
                Dichotomy refinement by resampling the cost surface (double data).

                :param cost_volumes: cost volumes data
                :type cost_volumes: NDArray[np.floating]
                :param disparity_map_col: column disparity map data
                :type disparity_map_col: NDArray[np.floating]
                :param disparity_map_row: row disparity map data
                :type disparity_map_row: NDArray[np.floating]
                :param score_map: score map data
                :type score_map: NDArray[np.floating]
                :param invalid_map: invalid map data
                :type invalid_map: NDArray[np.uint8]
                :param criteria_map: criteria map data
                :type criteria_map: NDArray[np.floating]
                :param subpixel: sub-sampling of cost_volume
                :type subpixel: int
                :param nb_iterations: number of iterations of the dichotomy
                :type nb_iterations: int
                :param filter: interpolation filter
                :type filter: abstractfilter::AbstractFilter
                :param type_method_matching_cost: max or min
                :type type_method_matching_cost: str
            )mydelimiter");

  m.def("compute_dichotomy_image_float", &compute_dichotomy_image<P2d::Vectorf>, "nb_cols"_a,
        "left_image"_a, "right_image"_a, "disparity_map_col"_a, "disparity_map_row"_a,
        "score_map"_a, "invalid_map"_a.noconvert(), "subpixel"_a, "nb_iterations"_a, "filter"_a,
        "method_matching_cost"_a, "window_size"_a, "step"_a, "offset_cv_img"_a, "disp_min"_a,
        "right_valid_mask"_a,
        R"mydelimiter(
                Dichotomy refinement by resampling the right image (float data).

                :param nb_cols: number of columns in the cost volume
                :type nb_cols: int
                :param left_image: left image data
                :type left_image: NDArray[np.floating]
                :param right_image: right image data
                :type right_image: NDArray[np.floating]
                :param disparity_map_col: column disparity map data
                :type disparity_map_col: NDArray[np.floating]
                :param disparity_map_row: row disparity map data
                :type disparity_map_row: NDArray[np.floating]
                :param score_map: score map data
                :type score_map: NDArray[np.floating]
                :param invalid_map: invalid map data
                :type invalid_map: NDArray[np.uint8]
                :param subpixel: sub-sampling of cost_volume
                :type subpixel: int
                :param nb_iterations: number of iterations of the dichotomy
                :type nb_iterations: int
                :param filter: interpolation filter
                :type filter: abstractfilter::AbstractFilter
                :param method_matching_cost: matching cost method
                :type method_matching_cost: str
                :param window_size: size of the correlation window
                :type window_size: int
                :param step: step of the matching cost
                :type step: list[int]
                :param offset_cv_img: offset of the cost volume in the image
                :type offset_cv_img: list[int]
                :param disp_min: minimum disparity values
                :type disp_min: list[float]
                :param right_valid_mask: mask of valid pixels in the right image
                :type right_valid_mask: NDArray[np.uint8]
            )mydelimiter");

  m.def("compute_dichotomy_image_double", &compute_dichotomy_image<P2d::VectorD>, "nb_cols"_a,
        "left_image"_a, "right_image"_a, "disparity_map_col"_a, "disparity_map_row"_a,
        "score_map"_a, "invalid_map"_a.noconvert(), "subpixel"_a, "nb_iterations"_a, "filter"_a,
        "method_matching_cost"_a, "window_size"_a, "step"_a, "offset_cv_img"_a, "disp_min"_a,
        "right_valid_mask"_a,
        R"mydelimiter(
                Dichotomy refinement by resampling the right image (double data).

                :param nb_cols: number of columns in the cost volume
                :type nb_cols: int
                :param left_image: left image data
                :type left_image: NDArray[np.floating]
                :param right_image: right image data
                :type right_image: NDArray[np.floating]
                :param disparity_map_col: column disparity map data
                :type disparity_map_col: NDArray[np.floating]
                :param disparity_map_row: row disparity map data
                :type disparity_map_row: NDArray[np.floating]
                :param score_map: score map data
                :type score_map: NDArray[np.floating]
                :param invalid_map: invalid map data
                :type invalid_map: NDArray[np.uint8]
                :param subpixel: sub-sampling of cost_volume
                :type subpixel: int
                :param nb_iterations: number of iterations of the dichotomy
                :type nb_iterations: int
                :param filter: interpolation filter
                :type filter: abstractfilter::AbstractFilter
                :param method_matching_cost: matching cost method
                :type method_matching_cost: str
                :param window_size: size of the correlation window
                :type window_size: int
                :param step: step of the matching cost
                :type step: list[int]
                :param offset_cv_img: offset of the cost volume in the image
                :type offset_cv_img: list[int]
                :param disp_min: minimum disparity values
                :type disp_min: list[float]
                :param right_valid_mask: mask of valid pixels in the right image
                :type right_valid_mask: NDArray[np.uint8]
            )mydelimiter");
}