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
This module contains functions associated to the zncc in cpp.
*/

#ifndef ZNCC_HPP
#define ZNCC_HPP

#include "operation.hpp"
#include "pandora2d_type.hpp"

const double STD_EPSILON = 1e-8;  // is 1e-16 for the variance.

/**
 * @brief Compute zncc between two images
 *
 *
 * @param left_image left image
 * @param right_image right image
 * @return double zncc value
 */
template <typename T>
T calculate_zncc(const P2d::MatrixX<T>& left_image, const P2d::MatrixX<T>& right_image) {
  auto left_std_dev = std_dev(left_image);
  auto right_std_dev = std_dev(right_image);
  if (left_std_dev <= STD_EPSILON || right_std_dev <= STD_EPSILON) {
    return 0;
  }
  return ((left_image.array() - left_image.mean()) * (right_image.array() - right_image.mean()))
             .sum() /
         (left_std_dev * right_std_dev * left_image.size());
}

#endif