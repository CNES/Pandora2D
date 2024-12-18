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
This file contains useful function declarations for tests.
*/

#include "conftest.hpp"
#include "bin.hpp"
#include "operation.hpp"

#include <doctest.h>
#include <iostream>
#include <random>

/**
 * Check size and element on vector with a groundtruth
 */
void check_inside_eigen_element(Eigen::VectorXd data, Eigen::VectorXd expected) {
  REQUIRE(data.size() == expected.size());
  auto d = data.data();
  auto e = expected.data();
  for (; e != (expected.data() + expected.size()); ++d, ++e) {
    CHECK(*d == *e);
  }
}

Eigen::MatrixXd create_normal_matrix(std::size_t size, float mean, float std) {
  // random device class instance, source of 'true' randomness for initializing random seed
  std::random_device rd{};
  // Mersenne twister PRNG, initialized with seed from previous random device instance
  std::mt19937 gen{rd()};
  // instance of class std::normal_distribution with specific mean and stddev
  std::normal_distribution<float> dis{mean, std};

  // Create matrix
  std::vector<double> X(size * size);
  auto v_size = static_cast<int>(X.size());
  for (int i = 0; i < v_size; i++) {
    X[i] = dis(gen);
  }
  Eigen::Map<Eigen::MatrixXd> img(X.data(), size, size);

  return img;
}

/**
 * Create image
 */
Eigen::MatrixXd create_image(std::size_t size, float mean, float std, double nb_bins) {
  auto matrix = create_normal_matrix(size, mean, std);

  auto check_nb_bins = [](auto& matrix) -> double {
    auto h0 = get_bins_width(matrix);
    auto dynamique = matrix.maxCoeff() - matrix.minCoeff();
    return dynamique / h0;
  };

  if (check_nb_bins(matrix) >= nb_bins)
    return matrix;

  auto h0 = get_bins_width(matrix);
  auto new_dynamique = std::ceil(nb_bins * h0);

  auto elt = 2;
  for (auto m = matrix.data(); m < matrix.data() + elt; ++m) {
    *m = static_cast<double>(mean) - new_dynamique / 2.;
  }

  for (auto m = matrix.data() + elt; m < matrix.data() + 2 * elt; ++m) {
    *m = static_cast<double>(mean) + new_dynamique / 2.;
  }

  return matrix;
}