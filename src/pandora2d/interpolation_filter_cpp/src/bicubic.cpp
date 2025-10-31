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

#include "bicubic.hpp"

// Constructor with no parameter
Bicubic::Bicubic() {
  m_margins = Margins(1, 1, 2, 2);
}

// Get coefficients
P2d::VectorD Bicubic::get_coeffs(const double fractional_shift) {
  P2d::VectorD tab_coeffs(m_size);

  for (int i = 0; i < m_size; ++i) {
    double dist = std::abs(-1.0 + i - fractional_shift);
    if (dist <= 1.0) {
      tab_coeffs[i] = (((m_alpha + 2.0) * dist - (m_alpha + 3.0)) * dist * dist) + 1.0;
    } else {
      tab_coeffs[i] =
          (((m_alpha * dist - 5.0 * m_alpha) * dist) + 8.0 * m_alpha) * dist - 4.0 * m_alpha;
    }
  };

  return tab_coeffs;
};
