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
This module contains functions associated to the Abstract filter class for cpp.
*/

#ifndef INTERPOLATIONFILTER_HPP
#define INTERPOLATIONFILTER_HPP

#include "constant.hpp"
#include "margins.hpp"
#include "pandora2d_type.hpp"

namespace abstractfilter {

/**
 * @brief  This abstract class allows for the instantiation of a filter.
 */

class AbstractFilter {
 public:
  /**
   * @brief Construct a new Abstract Filter object
   *
   */
  AbstractFilter() = default;

  /**
   * @brief Construct a new Abstract Filter object
   *
   * @param size size of the filter
   * @param margins margins of the fiter
   */
  AbstractFilter(int size, Margins margins);

  /**
   * @brief Destroy the Abstract Filter object
   *
   */
  ~AbstractFilter() = default;

  /**
   * @brief Get the coeffs object
   *
   * @param fractional_shift positive fractional shift of the subpixel
   * position to be interpolated
   * @return P2d::VectorD, an array of interpolator coefficients
   * whose size depends on the filter margins
   */
  virtual P2d::VectorD get_coeffs(const double fractional_shift) = 0;

  /**
   * @brief  Returns the value of the interpolated position
   *
   * @param resampling_area area on which interpolator coefficients will be applied
   * @param row_coeff interpolator coefficients in cols
   * @param col_coeff interpolator coefficients in rows
   * @return double
   */
  double apply(const P2d::MatrixD& resampling_area,
               const P2d::VectorD& row_coeff,
               const P2d::VectorD& col_coeff) const;

  /**
   * @brief
   *
   * @param image image
   * @param col_positions subpix columns positions to be interpolated
   * @param row_positions subpix rows positions to be interpolated
   * @param max_fractional_value maximum fractional value used to get coefficients
   * @return P2d::VectorD, the interpolated value of the position
   * corresponding to col_coeff and row_coeff
   */
  P2d::VectorD interpolate(const P2d::MatrixD& image,
                           const P2d::VectorD& col_positions,
                           const P2d::VectorD& row_positions,
                           const double max_fractional_value = MAX_FRACTIONAL_VALUE);

  /**
   * @brief Get the size attribute
   *
   * @return int
   */
  int get_size() const { return m_size; }

  /**
   * @brief Get the margins attribute
   *
   * @return Margins
   */
  Margins get_margins() const { return m_margins; }

 protected:
  int m_size = 4;                 ///< filter size
  Margins m_margins{0, 0, 0, 0};  ///< filter margins
};
}  // namespace abstractfilter

#endif  // namespace filter