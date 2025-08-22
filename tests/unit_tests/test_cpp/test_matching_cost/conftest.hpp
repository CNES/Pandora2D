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
This file contains useful function definitions for matching_cost tests.
*/

#ifndef MATCHING_COST_CONFTEST_HPP
#define MATCHING_COST_CONFTEST_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "bin.hpp"
#include "cost_volume.hpp"
#include "global_conftest.hpp"
#include "pandora2d_type.hpp"

namespace py = pybind11;

/**
 * Create image
 */
template <typename T>
P2d::MatrixX<T> create_image(std::size_t size, T mean, T std, T nb_bins = 120) {
  auto matrix = create_normal_matrix<T>(size, mean, std);

  auto check_nb_bins = [](auto& matrix) -> T {
    auto h0 = get_bins_width(matrix);
    auto dynamic_range = matrix.maxCoeff() - matrix.minCoeff();
    return dynamic_range / h0;
  };

  if (check_nb_bins(matrix) >= nb_bins)
    return matrix;

  auto h0 = get_bins_width(matrix);
  auto new_dynamic_range = std::ceil(nb_bins * h0);

  auto elt = 2;
  for (auto m = matrix.data(); m < matrix.data() + elt; ++m) {
    *m = static_cast<T>(mean) - new_dynamic_range / 2.;
  }

  for (auto m = matrix.data() + elt; m < matrix.data() + 2 * elt; ++m) {
    *m = static_cast<T>(mean) + new_dynamic_range / 2.;
  }

  return matrix;
}

/**
 * @brief Load criteria dataarray saved as a binary file containing a 1D numpy array of type uint8
 *
 * @param filename: name of binary file containing the criteria array
 * @param cv_size: cost volumes size
 */
py::array_t<uint8_t> load_criteria_dataarray(const std::string& filename,
                                             const CostVolumeSize& cv_size);

/**
 * @brief Get data_path for matching cost test data
 *
 */

extern const char* data_path_env;
extern const std::string data_path;

#endif