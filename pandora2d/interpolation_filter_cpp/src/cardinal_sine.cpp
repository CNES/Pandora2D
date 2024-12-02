/* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
#include "cardinal_sine.hpp"

#include <algorithm>
#include <string>

Eigen::Index find_or_throw(const double value, const t_Vector &container,
                           const std::string &message)
{
  auto begin = container.begin();
  auto end = container.end();
  auto it = std::find(begin, end, value);
  if (it == end)
  {
    throw std::invalid_argument(message);
  }
  return std::distance(begin, it);
}

double sinc(double x)
{
  auto y = M_PI * x; // Starting with C++20, use std::numbers::pi instead
  return y != 0.0 ? std::sin(y) / y : 1.0;
}

t_Vector fractional_range(double fractional_shift)
{
  int size = std::ceil(1.0 / fractional_shift);
  return Eigen::VectorXd::LinSpaced(size, 0.0, (size - 1) * fractional_shift);
}

t_Matrix compute_coefficient_table(int filter_size,
                                   const t_Vector &fractional_shifts)
{
  auto sigma = filter_size;
  auto aux1 = (-2.0 * M_PI) / (sigma * sigma);
  t_Array coeff_range = t_Array::LinSpaced(2 * filter_size + 1, -filter_size, filter_size);

  // With a filter size of 6 and fractional_shifts {0., 0.25, 0.5, 0.75}, aux value is:
  // [[ 6.  ,  5.  ,  4.  ,  3.  ,  2.  ,  1.  ,  0.  , -1.  , -2.  ,
  //   -3.  , -4.  , -5.  , -6.  ],
  //  [ 6.25,  5.25,  4.25,  3.25,  2.25,  1.25,  0.25, -0.75, -1.75,
  //   -2.75, -3.75, -4.75, -5.75],
  //  [ 6.5 ,  5.5 ,  4.5 ,  3.5 ,  2.5 ,  1.5 ,  0.5 , -0.5 , -1.5 ,
  //   -2.5 , -3.5 , -4.5 , -5.5 ],
  //  [ 6.75,  5.75,  4.75,  3.75,  2.75,  1.75,  0.75, -0.25, -1.25,
  //   -2.25, -3.25, -4.25, -5.25],
  // ]
  t_Matrix aux =
      fractional_shifts.rowwise().replicate(coeff_range.size()).rowwise() -
      coeff_range.transpose().matrix();

  t_Matrix sinc_values = aux.unaryExpr(std::ref(sinc));

  // Compute coefficients
  t_Matrix table_coeff =
      sinc_values.array() *
      (aux1 * aux.array() * aux.array()).array().exp().array();

  // Normalize coefficients by rows sums.
  return table_coeff.array() / table_coeff.rowwise()
                                   .sum()
                                   .rowwise()
                                   .replicate(table_coeff.cols())
                                   .array();
}

CardinalSine::CardinalSine() : AbstractFilter{13, {6, 6, 6, 6}},
                               m_half_size{6},
                               m_fractional_shifts{fractional_range(0.25)},
                               m_coeffs{compute_coefficient_table(m_half_size, m_fractional_shifts)}
{
}

CardinalSine::CardinalSine(int half_size, double fractional_shift) :
    AbstractFilter{1 + 2 * half_size, {half_size, half_size, half_size, half_size}},
    m_half_size(half_size),
    m_fractional_shifts{fractional_range(fractional_shift)},
    m_coeffs{compute_coefficient_table(half_size, m_fractional_shifts)}
{
}

t_Vector CardinalSine::get_coeffs(const double fractional_shift)
{
  Eigen::Index index = find_or_throw(fractional_shift, m_fractional_shifts,
                                     "Unknown fractional shift.");
  return m_coeffs.row(index);
}
