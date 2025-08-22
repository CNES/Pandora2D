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
This file contains useful function definitions for tests.
*/

#ifndef CONFTEST_HPP
#define CONFTEST_HPP

#include <doctest.h>
#include <random>
#include "cost_volume.hpp"
#include "pandora2d_type.hpp"

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
 * @brief Create a normal matrix object
 */
template <typename T>
P2d::MatrixX<T> create_normal_matrix(std::size_t size, T mean, T std) {
  // random device class instance, source of 'true' randomness for initializing random seed
  std::random_device rd{};
  // Mersenne twister PRNG, initialized with seed from previous random device instance
  std::mt19937 gen{rd()};
  // instance of class std::normal_distribution with specific mean and stddev
  std::normal_distribution<T> dis{mean, std};

  // Create matrix
  std::vector<T> X(size * size);
  auto v_size = static_cast<int>(X.size());
  for (int i = 0; i < v_size; i++) {
    X[i] = dis(gen);
  }
  Eigen::Map<P2d::MatrixX<T>> img(X.data(), size, size);

  return img;
}

/**
 * @brief Returns 1D index according to 2D position
 *
 * @param pixel 2D position
 * @param cv_size cost volume size
 */
auto position2d_to_index(Position2D& pixel, CostVolumeSize& cv_size) -> unsigned int;

#endif