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
 * @return T entropy
 */
template <typename T, typename U>
T get_entropy(const T nb_pixel, const U& hist) {
  T entropy = 0.0;

  for (auto bin_value : hist.values().reshaped()) {
    if (bin_value != 0.) {
      entropy -= bin_value / nb_pixel * std::log2(bin_value / nb_pixel);
    }
  };

  // Entropy cannot be negative
  return entropy < 0. ? 0. : entropy;
};

/**
 * @brief Compute entropy 1D of an image
 *
 * Entropy1D(img) = - sum(bin_value/nb_pixel * log2(bin_value/nb_pixel))
 * for each bin_value in Hist1D(img)
 *
 * @param image
 * @return T entropy1D
 */
template <typename T>
T calculate_entropy1D(const P2d::MatrixX<T>& image) {
  auto nb_pixel = static_cast<T>(image.size());
  auto hist_1D = calculate_histogram1D<T>(image);

  return get_entropy<T, Histogram1D<T>>(nb_pixel, hist_1D);
};

/**
 * @brief Compute entropy 2D of two images
 *
 * Entropy2D(img_l, img_r) = - sum(bin_value/nb_pixel * log2(bin_value/nb_pixel))
 * for each bin_value in Hist2D(img_l, img_r)
 *
 * @param left_image left image
 * @param right_image right image
 * @return T entropy 2D
 */
template <typename T>
T calculate_entropy2D(const P2d::MatrixX<T>& left_image, const P2d::MatrixX<T>& right_image) {
  // same size for left and right images
  auto nb_pixel = static_cast<T>(left_image.size());
  auto hist_2D = calculate_histogram2D<T>(left_image, right_image);

  return get_entropy<T, Histogram2D<T>>(nb_pixel, hist_2D);
};

/**
 * @brief Compute mutual information between two images
 *
 * MutualInformation(img_l,img_r) = Entropy1D(img_l) + Entropy1D(img_r) - Entropy2D(img_l, img_r)
 *
 * @param left_image left image
 * @param right_image right image
 * @return T mutual information value
 */
template <typename T>
T calculate_mutual_information(const P2d::MatrixX<T>& left_image,
                               const P2d::MatrixX<T>& right_image) {
  T mutual_information = calculate_entropy1D<T>(left_image) + calculate_entropy1D<T>(right_image) -
                         calculate_entropy2D<T>(left_image, right_image);

  return mutual_information;
}

#endif