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

/*
This module contains functions associated to the Abstract filter class for cpp.
*/

#include "interpolation_filter.hpp"
#include <limits>
#include <cmath>
#include <utility>

namespace abstractfilter
{

    // Constructor
    AbstractFilter::AbstractFilter(int size = 4, Margins margins = {0, 0, 0, 0}) :
        m_size(size),
        m_margins(margins)
    {
    }

    t_Vector AbstractFilter::get_coeffs(const double fractional_shift)
    {
        return t_Vector();
    }

    // Apply
    double AbstractFilter::apply(const t_Matrix &resampling_area,
                                 const t_Vector &row_coeff,
                                 const t_Vector &col_coeff) const
    {

        t_Vector intermediate_result = resampling_area * col_coeff;
        return row_coeff.dot(intermediate_result);
    }

    // Interpolate
    t_Vector AbstractFilter::interpolate(const t_Matrix &image,
                                         const t_Vector &col_positions,
                                         const t_Vector &row_positions,
                                         const double max_fractional_value)
    {

        // Initialisation of the result list
        t_Vector interpolated_positions = t_Vector::Zero(col_positions.size());

        // AbstractFilter
        const Margins &my_margins = AbstractFilter::m_margins;
        const int filter_size = AbstractFilter::m_size;

        auto col_it = col_positions.begin();
        auto row_it = row_positions.begin();
        auto result_it = interpolated_positions.begin();

        for (; col_it != col_positions.end(); ++col_it, ++row_it, ++result_it)
        {
            // get_coeffs method receives positive coefficients
            double fractional_row = std::abs(*row_it - std::floor(*row_it));
            double fractional_col = std::abs(*col_it - std::floor(*col_it));

            // If the subpixel shift is too close to 1, max_fractional_value is returned
            // to avoid rounding.
            if (1 - fractional_row < EPSILON)
            {
                fractional_row = max_fractional_value;
            }
            if (1 - fractional_col < EPSILON)
            {
                fractional_col = max_fractional_value;
            }

            // Get interpolation coefficients for fractional_row and fractional_col shifts
            t_Vector coeffs_row = this->get_coeffs(fractional_row);
            t_Vector coeffs_col = this->get_coeffs(fractional_col);

            /*
            Computation of the top left point of the resampling area
            on which the interpolating coefficients will be applied with apply method
            In cost_surface, row dimension is disp_col and column dimension is disp_row,
            then we use margins.left for row and margins.up for col
            */
            int top_left_area_row = *row_it - my_margins.left;
            int top_left_area_col = *col_it - my_margins.up;

            // Resampling area to which we will apply the interpolator coefficients
            t_Matrix resampling_area = image.block(top_left_area_row, top_left_area_col,
                                                   filter_size, filter_size);

            // Application of the interpolator coefficients on resampling area
            const auto result = apply(resampling_area, coeffs_row, coeffs_col);
            *result_it = result;
        }

        return interpolated_positions;
    }

} // namespace Abstractfilter
