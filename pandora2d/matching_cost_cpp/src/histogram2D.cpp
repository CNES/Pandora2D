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
This module contains functions associated to histogram.
*/

#include "histogram2D.hpp"

/**
 * @brief Construct a new Histogram 2D
 *
 * @param values: on histogram
 * @param row_hist: Histogram1D
 * @param col_hist: Histogram1D
 */
Histogram2D::Histogram2D(P2d::MatrixD& values, Histogram1D row_hist, Histogram1D col_hist)
    : m_values(values), m_row_hist(row_hist), m_col_hist(col_hist) {}

/**
 * @brief Construct a new Histogram 2D
 *
 * @param row_hist: Histogram1D
 * @param col_hist: Histogram1D
 */
Histogram2D::Histogram2D(Histogram1D row_hist, Histogram1D col_hist)
    : m_row_hist(row_hist), m_col_hist(col_hist) {
  m_values.setZero(m_row_hist.nb_bins(), m_col_hist.nb_bins());
}

/**
 * @brief Create and compute Histogram 2D
 *
 * @param img_l: left image
 * @param img_r: right image
 * @return Histogram2D
 */
Histogram2D calculate_histogram2D(const P2d::MatrixD& img_l, const P2d::MatrixD& img_r) {
  auto hist_l = Histogram1D(img_l);
  auto hist_r = Histogram1D(img_r);
  P2d::MatrixD values = P2d::MatrixD::Zero(hist_l.nb_bins(), hist_r.nb_bins());
  auto pixel_l = img_l.data();
  auto pixel_r = img_r.data();
  auto nb_bins_l = static_cast<int>(hist_l.nb_bins());
  auto nb_bins_r = static_cast<int>(hist_r.nb_bins());
  for (; pixel_l != (img_l.data() + img_l.size()); ++pixel_l, ++pixel_r) {
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