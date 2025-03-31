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
This module contains functions associated to the Bicubic filter class for cpp.
*/

#ifndef BICUBIC_HPP
#define BICUBIC_HPP

#include "interpolation_filter.hpp"

/**
 * @brief This struct allows for the instantiation of a bicubic filter.
 *
 *
 */
struct Bicubic : public abstractfilter::AbstractFilter {
  /**
   * @brief Construct a new Bicubic object
   *
   */
  Bicubic();

  /**
   * @brief Destroy the Bicubic object
   *
   */
  ~Bicubic() = default;

  /**
   * @brief Get the coeffs object
   *
   * @param fractional_shift positive fractional shift of the subpixel
   * position to be interpolated
   * @return P2d::VectorD, an array of interpolator coefficients
   * whose size depends on the filter margins
   */
  P2d::VectorD get_coeffs(const double fractional_shift) override;

  /**
   * @brief Get the alpha attribute
   *
   * @return float
   */
  float get_alpha() const { return m_alpha; }

 private:
  float m_alpha = -0.5;  ///< alpha coefficient
};

#endif