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

#include "mutual_information.hpp"

/**
 * @brief Compute entropy
 *
 * @tparam T hist1D or hist2D
 * @param nb_pixel of the image
 * @param hist to iterate
 * @return double entropy
 */
template <typename T>
double get_entropy(const double nb_pixel, const T& hist) {
  double entropy = 0.0;

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
 * @param img
 * @return double entropy1D
 */
double calculate_entropy1D(const P2d::MatrixD& img) {
  auto nb_pixel = static_cast<double>(img.size());
  auto hist_1D = calculate_histogram1D(img);

  return get_entropy<Histogram1D>(nb_pixel, hist_1D);
};

/**
 * @brief Compute entropy 2D of two images
 *
 * Entropy2D(img_l, img_r) = - sum(bin_value/nb_pixel * log2(bin_value/nb_pixel))
 * for each bin_value in Hist2D(img_l, img_r)
 *
 * @param img_l left image
 * @param img_r right image
 * @return double entropy 2D
 */
double calculate_entropy2D(const P2d::MatrixD& img_l, const P2d::MatrixD& img_r) {
  // same size for left and right images
  auto nb_pixel = static_cast<double>(img_l.size());
  auto hist_2D = calculate_histogram2D(img_l, img_r);

  return get_entropy<Histogram2D>(nb_pixel, hist_2D);
};

/**
 * @brief Compute mutual information between two images
 *
 * MutualInformation(img_l,img_r) = Entropy1D(img_l) + Entropy1D(img_r) - Entropy2D(img_l, img_r)
 *
 * @param img_l left image
 * @param img_r right image
 * @return double mutual information value
 */
double calculate_mutual_information(const P2d::MatrixD& img_l, const P2d::MatrixD& img_r) {
  double mutual_information =
      calculate_entropy1D(img_l) + calculate_entropy1D(img_r) - calculate_entropy2D(img_l, img_r);

  return mutual_information;
}
