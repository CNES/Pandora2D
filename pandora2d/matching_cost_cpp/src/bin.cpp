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

#include "bin.hpp"

/**
 * @brief Scott method to compute bin width
 * @param img : the Eigen matrix
 *
 */
double get_bins_width_scott(const P2d::MatrixD& img) {
  auto standard_deviation = std_dev(img);
  if (standard_deviation == 0.)
    return 1.;

  double size = static_cast<double>(img.size());
  return SCOTT_FACTOR * standard_deviation * pow(size, -1. / 3.);
}

/**
 * Get bin width depending on bin_method
 * @param img : the Eigen matrix
 * @param method : the bin_method, default is scott
 *
 * @throws std::invalid_argument if provided method is not known
 */
double get_bins_width(const P2d::MatrixD& img, bin_method method) {
  switch (method) {
    case bin_method::scott:
      return get_bins_width_scott(img);
    default:
      throw std::invalid_argument("method to compute bins width does not exist");
  }
}