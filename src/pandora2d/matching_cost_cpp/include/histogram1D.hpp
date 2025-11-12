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

#ifndef HISTOGRAM1D_HPP
#define HISTOGRAM1D_HPP

#include "bin.hpp"
#include "pandora2d_type.hpp"

/**
 * @brief Instanciation of Histogram on 1D (one dimension)
 *
 */
template <typename T>
class Histogram1D {
 public:
  /**
   * @brief Construct a new Histogram 1D
   *
   * @param values: on histogram
   * @param nb_bins: number of bins
   * @param low_bound: smallest value on histogram
   * @param bins_width: size of one bin on histogram
   */
  Histogram1D(P2d::VectorX<T>& values, std::size_t nb_bins, T low_bound, T bins_width)
      : m_values(values), m_nb_bins(nb_bins), m_low_bound(low_bound), m_bins_width(bins_width) {}

  /**
   * @brief Construct a new Histogram 1D
   *
   * @param nb_bins: number of bins
   * @param low_bound: smallest value on histogram
   * @param bins_width: size of one bin on histogram
   */
  Histogram1D(std::size_t nb_bins, T low_bound, T bins_width)
      : m_nb_bins(nb_bins), m_low_bound(low_bound), m_bins_width(bins_width) {
    m_values = P2d::VectorX<T>::Zero(m_nb_bins);
  }
  /**
   * @brief Construct a new Histogram 1D
   *
   * @param image
   */
  Histogram1D(const P2d::MatrixX<T>& image) {
    create(image);
    m_values = P2d::VectorX<T>::Zero(m_nb_bins);
  }

  /**
   * @brief Destroy the Histogram1D object
   *
   */
  ~Histogram1D() = default;

  /**
   * @brief Get the Values object
   *
   * @return const P2d::VectorD&
   */
  const P2d::VectorX<T>& values() const { return m_values; };

  /**
   * @brief Get the Nb Bins object
   *
   * @return std::size_t
   */
  std::size_t nb_bins() const { return m_nb_bins; };

  /**
   * @brief Get the Low Bound object
   *
   * @return T
   */
  T low_bound() const { return m_low_bound; };

  /**
   * @brief Get the Up Bound object
   *
   * @return T
   */
  T up_bound() const { return m_low_bound + m_nb_bins * m_bins_width; };

  /**
   * @brief Get the Bin Width object
   *
   * @return T
   */
  T bins_width() const { return m_bins_width; };

  /**
   * @brief Set the Values object
   *
   * @param values
   */
  void set_values(const P2d::VectorX<T>& values) { m_values = values; };

 private:
  /**
   * @brief Create Histogram 1D object without compute values
   *
   * @param image
   */
  void create(const P2d::MatrixX<T>& image) {
    m_bins_width = get_bins_width(image);
    T min_coeff = image.minCoeff();
    T max_coeff = image.maxCoeff();
    T dynamic_range = max_coeff - min_coeff;
    m_nb_bins = static_cast<int>(1. + (dynamic_range / m_bins_width));

    // check nb_bins > NB_BINS_MAX
    if (m_nb_bins > NB_BINS_MAX) {
      m_nb_bins = NB_BINS_MAX;
      T moment = variance(image);
      max_coeff = std::min(static_cast<T>(4.) * moment, max_coeff);
      min_coeff = std::max(static_cast<T>(-4.) * moment, min_coeff);
      dynamic_range = max_coeff - min_coeff;
    }

    m_low_bound = min_coeff - (static_cast<T>(m_nb_bins) * m_bins_width - dynamic_range) / 2.;
  };

  P2d::VectorX<T> m_values;  ///< values on histogram
  std::size_t m_nb_bins;     ///< number of bins
  T m_low_bound;             ///< smaller value on histogram
  T m_bins_width;            ///< size of one bin
};

/**
 * @brief Create and compute Histogram 1D
 *
 * @param image
 * @return Histogram1D
 */
template <typename T>
Histogram1D<T> calculate_histogram1D(const P2d::MatrixX<T>& image) {
  auto hist = Histogram1D<T>(image);
  P2d::VectorX<T> hist_values = P2d::VectorX<T>::Zero(hist.nb_bins());
  auto low_bound = hist.low_bound();
  auto bin_width = hist.bins_width();
  auto nb_bins = static_cast<int>(hist.nb_bins());
  for (auto pixel : image.reshaped()) {
    // if we are in the NB_BINS_MAX case, some elements may be outside of the low_bound and the
    // up_bound
    auto index =
        std::max(0, std::min(static_cast<int>((pixel - low_bound) / bin_width), nb_bins - 1));
    hist_values[index] += 1;
  }

  hist.set_values(hist_values);
  return hist;
}

#endif