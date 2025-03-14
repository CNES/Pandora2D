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

#include "operation.hpp"

/**
 * @brief Standard deviation with Eigen matrix
 * @param m : the Eigen matrix
 *
 */
double std_dev(const P2d::MatrixD& m) {
  return sqrt((m.array() - m.mean()).square().sum() / (m.size()));
}

/**
 * @brief Variance with Eigen matrix
 * Method used to center the histogram
 * Link to Medicis code:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * random_var_d.c#L588
 *
 * @param m : the Eigen matrix
 *
 */
double variance(const P2d::MatrixD& m) {
  double moment1 = (m.array().sum()) / (m.size());
  double moment2 = (m.array().square().sum()) / (m.size());
  return moment2 - (moment1 * moment1);
}

/**
 * @brief Function to find the index of the minimum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmin(const P2d::VectorD& vec) {
  int min_index = -1;
  double min_value = std::numeric_limits<double>::infinity();
  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isnan(vec[i]) && vec[i] < min_value) {
      min_value = vec[i];
      min_index = i;
    }
  }
  return min_index;
}

/**
 * @brief Function to find the index of the maximum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmax(const P2d::VectorD& vec) {
  int max_index = -1;
  double max_value = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isnan(vec[i]) && vec[i] > max_value) {
      max_value = vec[i];
      max_index = i;
    }
  }
  return max_index;
}

/**
 * @brief Check if all the elements are the same
 *
 * @param data
 * @return true : all elements are the same
 * @return false : not all elements are the same
 */
bool all_same(const P2d::VectorD& data) {
  auto value_tested = *(data.begin());
  for (auto d : data) {
    if (d != value_tested)
      return false;
  }
  return true;
}