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

#ifndef INTERPOLATIONFILTER_HPP
#define INTERPOLATIONFILTER_HPP

#include "alias.hpp"
#include "margins.hpp"
#include <Eigen/Dense>

namespace abstractfilter
{

    /**
     * @brief  This abstract class allows for the instantiation of a filter.
     */

    class AbstractFilter
    {
    public:
        /**
         * @brief Construct a new Abstract Filter object
         *
         */
        AbstractFilter();
        /**
         * @brief Destroy the Abstract Filter object
         *
         */
        virtual ~AbstractFilter();

        /**
         * @brief Get the coeffs object
         *
         * @param fractional_shift positive fractional shift of the subpixel
         * position to be interpolated
         * @return t_Vector, an array of interpolator coefficients
         * whose size depends on the filter margins
         */
        virtual t_Vector get_coeffs(const double fractional_shift) = 0;

        /**
         * @brief  Returns the value of the interpolated position
         *
         * @param resampling_area area on which interpolator coefficients will be applied
         * @param row_coeff interpolator coefficients in cols
         * @param col_coeff interpolator coefficients in rows
         * @return double
         */
        double apply(const t_Matrix &resampling_area,
                     const t_Vector &row_coeff,
                     const t_Vector &col_coeff) const;

        /**
         * @brief
         *
         * @param image image
         * @param col_positions subpix columns positions to be interpolated
         * @param row_positions subpix rows positions to be interpolated
         * @param max_fractional_value maximum fractional value used to get coefficients
         * @return t_Vector, the interpolated value of the position
         * corresponding to col_coeff and row_coeff
         */
        t_Vector interpolate(const t_Matrix &image,
                             const t_Vector &col_positions,
                             const t_Vector &row_positions,
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
        int m_size = 4;
        Margins m_margins{0, 0, 0, 0};
    };
}

#endif // namespace filter