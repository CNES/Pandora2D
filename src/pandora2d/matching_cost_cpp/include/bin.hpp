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
  float sum = 0;
  float sum_sq = 0;

  // Use de-referenced pointer to read matrix element
  const float *value;
  
  int idx;
  T num_elem = static_cast<T>(image.size());

  // Compute variance according to the formula: E(X^2) - E(X)^2
  // The sum is computed in float as the input image
  for (idx = 0, value = &image(0); idx < image.size(); ++idx) {
    sum += *value;
    sum_sq += *value * *value;
    value++;
  }

  // Then we cast to T type to keep or increase precision (float32/64)
  T mean = static_cast<T>(sum) / num_elem;
  
  T standard_deviation = static_cast<T>(sum_sq) / num_elem - (mean * mean);
  if (standard_deviation == 0.)
    return 1.;

  T pow_size = static_cast<T>(std::pow(static_cast<double>(num_elem), -1. / 3.));
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
