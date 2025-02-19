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

#include "operation.hpp"

/**
 * @brief Standard deviation with Eigen matrix
 * @param m : the Eigen matrix
 *
 */
double std_dev(const Eigen::MatrixXd& m) {
  return sqrt((m.array() - m.mean()).square().sum() / (m.size()));
}

/**
 * @brief Variance with Eigen matrix
 * Method used to center the histogram
 * Link to Medicis code:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * random_var_d.c#L588
 *
 * @param m : the Eigen matrix
 *
 */
double variance(const Eigen::MatrixXd& m) {
  double moment1 = (m.array().sum()) / (m.size());
  double moment2 = (m.array().square().sum()) / (m.size());
  return moment2 - (moment1 * moment1);
}