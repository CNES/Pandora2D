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
class Histogram2D {
 public:
  /**
   * @brief Construct a new Histogram2D object
   * @param values: on histogram
   * @param row_hist: Histogram1D
   * @param col_hist: Histogram1D
   */
  Histogram2D(P2d::MatrixD& values, Histogram1D row_hist, Histogram1D col_hist);

  /**
   * @brief Construct a new Histogram2D object
   * @param row_hist: Histogram1D
   * @param col_hist: Histogram1D
   *
   */
  Histogram2D(Histogram1D row_hist, Histogram1D col_hist);

  /**
   * @brief Destroy the Histogram2D object
   *
   */
  ~Histogram2D() = default;

  /**
   * @brief Get the Values object
   *
   * @return const P2d::MatrixD&
   */
  const P2d::MatrixD& values() const { return m_values; };

  /**
   * @brief Set the Values object
   *
   * @param values
   */
  void set_values(const P2d::MatrixD& values) { m_values = values; };

 private:
  P2d::MatrixD m_values;   ///< values on histogram
  Histogram1D m_row_hist;  ///< row dimension (number of bins, size of bin, low bound)
  Histogram1D m_col_hist;  ///< col dimension (number of bins, size of bin, low bound)
};

/**
 * @brief Compute histogram 2D based on two images
 *
 * @param img_l : left image
 * @param img_r : right image
 * @return Histogram2D
 */
Histogram2D calculate_histogram2D(const P2d::MatrixD& img_l, const P2d::MatrixD& img_r);

#endif