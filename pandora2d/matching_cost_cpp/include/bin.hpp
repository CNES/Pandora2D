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
This module contains functions associated to bin (width, number).
*/

#ifndef BIN_HPP
#define BIN_HPP

#include "operation.hpp"
#include "pandora2d_type.hpp"

constexpr unsigned int NB_BINS_MAX = 100;  ///< Limit of number bins for histogram
constexpr double SCOTT_FACTOR = 3.491;     ///< factor for scott formula

/**
 * @brief All methods to compute the bin width
 *
 */
typedef enum bin_method {
  scott,  ///< Scott method https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/ScottBandwidth.pdf
} bin_method;

/**
 * @brief Scott method to compute bin width
 * @param img : the Eigen matrix
 *
 */
double get_bins_width_scott(const P2d::MatrixD& img);

/**
 * @brief Get bin width depending on bin_method
 * @param img : the Eigen matrix
 * @param method : the bin_method, default is scott
 *
 * @throws std::invalid_argument if provided method is not known
 */
double get_bins_width(const P2d::MatrixD& img, bin_method method = bin_method::scott);

#endif