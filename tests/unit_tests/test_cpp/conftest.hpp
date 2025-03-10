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
This file contains useful function definitions for tests.
*/

#ifndef CONFTEST_HPP
#define CONFTEST_HPP

#include <doctest.h>
#include "eigen_type.hpp"

/**
 * @brief Check size and element on vector with a groundtruth
 *
 * @param data: (vector or matrix) to test
 * @param expected: the groundtruth
 */
template <typename T>
void check_inside_eigen_element(const T& data, const T& expected) {
  REQUIRE(data.size() == expected.size());
  auto d = data.data();
  auto e = expected.data();
  for (; e != (expected.data() + expected.size()); ++d, ++e) {
    CHECK(*d == doctest::Approx(*e).epsilon(1e-6));
  }
}

/**
 * @brief Create normal matrix for test
 *
 * @param size: image shape
 * @param mean: image mean
 * @param std: image standard deviation
 */
t_MatrixD create_normal_matrix(std::size_t size, float mean, float std);

/**
 * @brief Create an image for tests
 *
 * @param size: image shape
 * @param mean: image mean
 * @param std: image standard deviation
 * @param nb_bins: bins number for image histogram
 */
t_MatrixD create_image(std::size_t size, float mean, float std, double nb_bins = 120);

#endif