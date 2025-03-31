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
This module contains functions associated to the Cardinal Sine filter class for
cpp.
*/

#ifndef CARDINALSINEFILTER_HPP
#define CARDINALSINEFILTER_HPP

#define _USE_MATH_DEFINES  // to use cmath and get M_PI
#include <cmath>

#include "interpolation_filter.hpp"

/**
 * @brief Find index of a value in a container or throw if not found.
 *
 * @param value value to search for.
 * @param container container to search in.
 * @param message message of the error.
 *
 * @throws std::invalid_argument if provided value is not found in container.
 *
 * @return Eigen::Index
 */
Eigen::Index find_or_throw(const double value,
                           const P2d::VectorD& container,
                           const std::string& message);

/**
 * @brief Compute Cardinal Sine of x (with PI).
 *
 * @param x value to compute cardinal sine from.
 * @return double cardinal sine of x.
 */
double sinc(double x);

/**
 * @brief Contruct a vector of fractional shifts in interval [0,1[ with a step of fractional shift.
 *
 * @param fractional_shift
 * @return P2d::VectorD
 */
P2d::VectorD fractional_range(double fractional_shift);

/**
 * @brief Compute normalized cardinal sine coefficients windowed by a Gaussian.
 *
 *   Will compute the `2 * filter_size + 1` coefficients for each given
 * fractional_shift in `fractional_shifts` and store them in the returned
 * NDArray where:
 *
 *       - Each row corresponds to a specific fractional shift value.
 *       - Each column corresponds to a coefficient at a specific position.
 *
 *   The Gaussian window width correspond to the size of the filter.
 * @param filter_size size of the filter.
 * @param fractional_shifts fractional shifts where to compute coefficients.
 * @return P2d::MatrixD (2*filter_size + 1) coefficients for each fractional shift.
 */
P2d::MatrixD compute_coefficient_table(int filter_size, const P2d::VectorD& fractional_shifts);

/**
 * @brief Cardinal sine filter
 *
 */
struct CardinalSine : public abstractfilter::AbstractFilter {
  /**
   * @brief Construct a new CardinalSine object
   *
   */
  CardinalSine();

  /**
   * @brief Construct a new Cardinal Sine object
   *
   * @param half_size half filter size.
   * @param fractional_shift interval between each interpolated point,
   * sometimes referred to as precision. Expected value in the range [0,1[.
   */
  CardinalSine(int half_size, double fractional_shift);

  /**
   * @brief Destroy the CardinalSine object
   *
   */
  ~CardinalSine() = default;

  /**
   * @brief Get the coeffs object
   *
   * @param fractional_shift positive fractional shift of the subpixel
   * position to be interpolated
   *
   * @throws std::invalid_argument if provided fractional_shift is not found
   * precomputed table.
   *
   * @return P2d::VectorD, an array of interpolator coefficients
   * whose size depends on the filter margins
   */
  P2d::VectorD get_coeffs(const double fractional_shift) override;

 private:
  int m_half_size;                   ///< Half filter size
  P2d::VectorD m_fractional_shifts;  ///< Fractional shifts used to compute coefficients
  P2d::MatrixD m_coeffs;             ///< Pre-computed coefficients
};
#endif