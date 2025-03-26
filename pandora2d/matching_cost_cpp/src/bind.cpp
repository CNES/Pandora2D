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
This module contains functions associated to the binding pybind of cpp cost volumes computation.
*/

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compute_cost_volumes.hpp"

using namespace pybind11::literals;

PYBIND11_MODULE(matching_cost_bind, m) {
  m.def("compute_cost_volumes_cpp", &compute_cost_volumes_cpp, "left"_a, "right"_a, "cv_values"_a,
        "cv_size"_a, "disp_range_row"_a, "disp_range_col"_a, "offset_cv_img_row"_a,
        "offset_cv_img_col"_a, "window_size"_a, "step"_a, "no_data"_a,
        R"mydelimiter(
            Computes the cost values

            :param left: left image
            :type left: NDArray[np.float64]
            :param right: list of right images
            :type right: List[NDArray[np.float64]]
            :param cv_values:  cost volumes initialized values
            :type cv_values: NDArray[np.float64]
            :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
            :type cv_size: CostVolumeSize
            :param disp_range_row:  cost volumes row disparity range
            :type disp_range_row: NDArray[np.float64]
            :param disp_range_col:  cost volumes col disparity range
            :type disp_range_col: NDArray[np.float64]
            :param offset_cv_img_row: row offset between first index of cv and image (ROI case)
            :type offset_cv_img_row: int
            :param offset_cv_img_col: col offset between first index of cv and image (ROI case)
            :type offset_cv_img_col: int
            :param window_size: size of the correlation window
            :type window_size: int
            :param step: [step_row, step_col]
            :type step: NDArray[np.integer]
            :param no_data: no data value in img
            :type no_data: float
            )mydelimiter");
}