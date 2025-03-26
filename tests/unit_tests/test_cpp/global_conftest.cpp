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
This file contains useful function declarations for tests.
*/

#include "global_conftest.hpp"

#include <random>

/**
 * @brief Create a normal matrix object
 */
P2d::MatrixD create_normal_matrix(std::size_t size, float mean, float std) {
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
  Eigen::Map<P2d::MatrixD> img(X.data(), size, size);

  return img;
}