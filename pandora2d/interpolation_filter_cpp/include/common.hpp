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
This module contains type and struct associated to the Abstract filter class for cpp.
*/

#ifndef INTERPOLATION_FILTER_COMMON_HPP
#define INTERPOLATION_FILTER_COMMON_HPP

#include <Eigen/Dense>

// Alias for eigen type
using t_Matrix = Eigen::MatrixXd;
using t_Vector = Eigen::VectorXd;

// Margins
/**
 * @brief 
 * 
 * 
 */
struct Margins
{
    /**
     * @brief Construct a new Margins object
     * 
     * @param left_ 
     * @param up_ 
     * @param right_ 
     * @param down_ 
     */
    Margins(int left_, int up_, int right_, int down_)
        : left(left_), up(up_), right(right_), down(down_) {}

    int left, up, right, down;
};

#endif