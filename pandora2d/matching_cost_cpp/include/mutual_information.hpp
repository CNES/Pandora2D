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
This module contains functions associated to the mutual information in cpp.
*/

#ifndef MUTUAL_INFORMATION_HPP
#define MUTUAL_INFORMATION_HPP

#include "histogram1D.hpp"
#include "histogram2D.hpp"
#include "pandora2d_type.hpp"

/**
 * @brief Compute entropy
 *
 * @tparam T hist1D or hist2D
 * @param nb_pixel of the image
 * @param hist to iterate
 * @return double entropy
 */
template <typename T>
double get_entropy(const double nb_pixel, const T& hist);

/**
 * @brief Compute entropy 1D of an image
 *
 * Entropy1D(img) = - sum(bin_value/nb_pixel * log2(bin_value/nb_pixel))
 * for each bin_value in Hist1D(img)
 *
 * @param image
 * @return double entropy1D value
 */
double calculate_entropy1D(const P2d::MatrixD& image);

/**
 * @brief Compute entropy 2D of two images
 *
 * Entropy2D(img_l, img_r) = - sum(bin_value/nb_pixel * log2(bin_value/nb_pixel))
 * for each bin_value in Hist2D(img_l, img_r)
 *
 * @param left_image left image
 * @param right_image right image
 * @return double entropy 2D value
 */
double calculate_entropy2D(const P2d::MatrixD& left_image, const P2d::MatrixD& right_image);

/**
 * @brief Compute mutual information between two images
 *
 * MutualInformation(img_l,img_r) = Entropy1D(img_l) + Entropy1D(img_r) - Entropy2D(img_l, img_r)
 *
 * @param left_image left image
 * @param right_image right image
 * @return double mutual information value
 */
template <typename T>
T calculate_mutual_information(const P2d::MatrixX<T>& left_image,
                               const P2d::MatrixX<T>& right_image) {
  double mutual_information =
      calculate_entropy1D(left_image.template cast<double>()) +
      calculate_entropy1D(right_image.template cast<double>()) -
      calculate_entropy2D(left_image.template cast<double>(), right_image.template cast<double>());

  return static_cast<T>(mutual_information);
}

#endif