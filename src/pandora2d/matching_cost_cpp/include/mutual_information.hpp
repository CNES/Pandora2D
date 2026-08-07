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

#include <optional>

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
 * @brief Mutual information correlator
 *
 * MutualInformation(img_l,img_r) = Entropy1D(img_l) + Entropy1D(img_r) - Entropy2D(img_l, img_r)
 *
 */
template <typename T>
struct MutualInformationCorrelator {
  // Pointer to the current left window
  const P2d::Matrixf* left_window = nullptr;
  // We use std::optional because Histogram1D does not have a default constructor
  std::optional<Histogram1D<T>> hist_left;
  // Initialized entropy of the left window
  T entropy_left{};
  // Initialized number of pixels in the windows
  T nb_pixel{};

  /**
   * @brief Prepare the mutual information correlator for the left window.
   *
   * This function calculates and stores the histogram and entropy of the left window.
   *
   * @param new_left_window The new left image window
   */
  void prepare_left_window(const P2d::Matrixf& new_left_window) {
    left_window = &new_left_window;
    hist_left = calculate_histogram1D<T>(new_left_window);
    nb_pixel = static_cast<T>(new_left_window.size());
    entropy_left = get_entropy<T, Histogram1D<T>>(nb_pixel, *hist_left);
  }

  /**
   * @brief Compute the mutual information between the prepared left window and a right window.
   *
   * This function calculates the mutual information using the stored histogram and entropy of the
   * left window and computes the histogram and entropy of the right window.
   *
   * MutualInformation(img_l,img_r) = Entropy1D(img_l) + Entropy1D(img_r) - Entropy2D(img_l, img_r)
   *
   * @param right_window The right image window
   * @return T The mutual information value
   */
  T operator()(const P2d::Matrixf& right_window) const {
    Histogram1D<T> hist_right = calculate_histogram1D<T>(right_window);
    T entropy_right = get_entropy<T, Histogram1D<T>>(nb_pixel, hist_right);

    auto hist_2d = calculate_histogram2D<T>(*left_window, right_window, *hist_left, hist_right);
    T entropy_2d = get_entropy<T, Histogram2D<T>>(nb_pixel, hist_2d);

    return entropy_left + entropy_right - entropy_2d;
  }
};

#endif
