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
This module contains functions, struct and other elements associated to the matching_cost.
*/

#ifndef OPERATION_HPP
#define OPERATION_HPP

#include <Eigen/Dense>

/**
 * @brief Standard deviation with Eigen matrix
 * @param m : the Eigen matrix
 * 
 */
double std_dev(const Eigen::MatrixXd &m);


/**
 * @brief Method of centring the histogram
 * Link to Medicis code:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * random_var_d.c#L588
 *
 * @param m : the Eigen matrix
 * 
 */
double moment_centre(const Eigen::MatrixXd &m);

#endif

