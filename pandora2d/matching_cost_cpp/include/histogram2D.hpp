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
This module contains functions associated to the histogram for cpp.
*/

#ifndef HISTOGRAM2D_HPP
#define HISTOGRAM2D_HPP

#include "histogram1D.hpp"
#include "pandora2d_type.hpp"

/**
 * @brief Instanciation of Histogram on 2D (two dimension)
 *
 */
template <typename T>
class Histogram2D {
 public:
  /**
   * @brief Construct a new Histogram 2D
   *
   * @param values: on histogram
   * @param row_hist: Histogram1D
   * @param col_hist: Histogram1D
   */
  Histogram2D(P2d::MatrixX<T>& values, Histogram1D<T> row_hist, Histogram1D<T> col_hist)
      : m_values(values), m_row_hist(row_hist), m_col_hist(col_hist) {}

  /**
   * @brief Construct a new Histogram 2D
   *
   * @param row_hist: Histogram1D
   * @param col_hist: Histogram1D
   */
  Histogram2D(Histogram1D<T> row_hist, Histogram1D<T> col_hist)
      : m_row_hist(row_hist), m_col_hist(col_hist) {
    m_values.setZero(m_row_hist.nb_bins(), m_col_hist.nb_bins());
  }

  /**
   * @brief Destroy the Histogram2D object
   *
   */
  ~Histogram2D() = default;

  /**
   * @brief Get the Values object
   *
   * @return const P2d::MatrixX<T>&
   */
  const P2d::MatrixX<T>& values() const { return m_values; };

  /**
   * @brief Set the Values object
   *
   * @param values
   */
  void set_values(const P2d::MatrixX<T>& values) { m_values = values; };

 private:
  P2d::MatrixX<T> m_values;   ///< values on histogram
  Histogram1D<T> m_row_hist;  ///< row dimension (number of bins, size of bin, low bound)
  Histogram1D<T> m_col_hist;  ///< col dimension (number of bins, size of bin, low bound)
};

/**
 * @brief Compute histogram 2D based on two images
 *
 * @param left_image : left image
 * @param right_image : right image
 * @return Histogram2D
 */
template <typename T>
Histogram2D<T> calculate_histogram2D(const P2d::MatrixX<T>& left_image,
                                     const P2d::MatrixX<T>& right_image) {
  auto hist_l = Histogram1D<T>(left_image);
  auto hist_r = Histogram1D<T>(right_image);
  P2d::MatrixX<T> values = P2d::MatrixX<T>::Zero(hist_l.nb_bins(), hist_r.nb_bins());
  auto pixel_l = left_image.data();
  auto pixel_r = right_image.data();
  auto nb_bins_l = static_cast<int>(hist_l.nb_bins());
  auto nb_bins_r = static_cast<int>(hist_r.nb_bins());
  for (; pixel_l != (left_image.data() + left_image.size()); ++pixel_l, ++pixel_r) {
    auto index_l = std::max(
        0, std::min(static_cast<int>((*pixel_l - hist_l.low_bound()) / hist_l.bins_width()),
                    nb_bins_l - 1));
    auto index_r = std::max(
        0, std::min(static_cast<int>((*pixel_r - hist_r.low_bound()) / hist_r.bins_width()),
                    nb_bins_r - 1));
    values(index_l, index_r) += 1;
  }
  return Histogram2D(values, hist_l, hist_r);
}

#endif