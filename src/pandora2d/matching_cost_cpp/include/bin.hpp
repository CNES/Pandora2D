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
This module contains functions associated to bin (width, number).
*/

#ifndef BIN_HPP
#define BIN_HPP

#include "operation.hpp"
#include "pandora2d_type.hpp"

constexpr unsigned int NB_BINS_MAX = 100;  ///< Limit of number bins for histogram
constexpr double SCOTT_FACTOR = 3.491;     ///< factor for scott formula

/**
 * @brief All methods to compute the bin width
 *
 */
typedef enum bin_method {
  scott,  ///< Scott method https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/ScottBandwidth.pdf
} bin_method;

/**
 * @brief Scott method to compute bin width
 * @param image : the Eigen matrix
 *
 */
template <typename T>
T get_bins_width_scott(const P2d::Matrixf& image) {
  // Compute variance according to the formula: E(X^2) - E(X)^2
  // The sum is computed in float as the input image
  auto sum = image.sum();
  auto sum_sq = image.squaredNorm();

  // Then we cast to T type to keep or increase precision (float32/64)
  auto num_elem = static_cast<T>(image.size());
  T mean = static_cast<T>(sum) / num_elem;
  T standard_deviation = std::sqrt(static_cast<T>(sum_sq) / num_elem - (mean * mean));
  if (standard_deviation == 0.)
    return 1.;

  T pow_size = pow(num_elem, -1. / 3.);
  return static_cast<T>(SCOTT_FACTOR) * standard_deviation * pow_size;
}

/**
 * Get bin width depending on bin_method
 * @param image : the Eigen matrix
 * @param method : the bin_method, default is scott
 *
 * @throws std::invalid_argument if provided method is not known
 */
template <typename T>
T get_bins_width(const P2d::Matrixf& image, bin_method method = bin_method::scott) {
  switch (method) {
    case bin_method::scott:
      return get_bins_width_scott<T>(image);
    default:
      throw std::invalid_argument("method to compute bins width does not exist");
  }
}
#endif
