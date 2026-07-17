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

  for (auto bin_value : hist.values().template reshaped<Eigen::RowMajor>()) {
    if (bin_value != 0.) {
      entropy -= bin_value / nb_pixel * std::log2(bin_value / nb_pixel);
    }
  };

  // Entropy cannot be negative
  return entropy < 0. ? 0. : entropy;
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
T calculate_mutual_information(const P2d::Matrixf& left_image, const P2d::Matrixf& right_image) {
  auto nb_pixel = static_cast<T>(left_image.size());

  // We calculate the histograms to avoid allocating them twice in the entropy functions
  auto hist_left = calculate_histogram1D<T>(left_image);
  auto hist_right = calculate_histogram1D<T>(right_image);

  T entropy_l = get_entropy<T, Histogram1D<T>>(nb_pixel, hist_left);
  T entropy_r = get_entropy<T, Histogram1D<T>>(nb_pixel, hist_right);

  auto hist_2d = calculate_histogram2D<T>(left_image, right_image, hist_left, hist_right);
  T entropy_2d = get_entropy<T, Histogram2D<T>>(nb_pixel, hist_2d);

  return entropy_l + entropy_r - entropy_2d;
}

#endif
