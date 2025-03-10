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
This module contains tests associated to the operation functions define on operation.hpp file.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "operation.hpp"

#include <Eigen/Dense>

/**
 * @brief Standard deviation calculation medicis version
 * The standard deviation is calculated here:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * random_var_d.c#L433
 * The square root part is calculated here:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * random_var_d.c#L588
 *
 * @param m : the Eigen matrix
 */
double std_dev_medicis(const t_MatrixD& m) {
  return sqrt(variance(m));
}

TEST_CASE("standard deviation with null matrix") {
  t_MatrixD image(2, 4);

  image << 0, 0, 0, 0, 0, 0, 0, 0;

  auto standard_deviation = std_dev(image);
  CHECK(standard_deviation == 0);
}

TEST_CASE("standard deviation with VectorXd") {
  t_VectorD image(2, 4);

  image << 1, 2, 3;

  auto standard_deviation = std_dev(image);
  CHECK(standard_deviation == 0.5);
}

TEST_CASE("standard deviation with MatrixXd one line") {
  t_MatrixD image(1, 4);

  image << 1, 2, 3, 4;

  auto standard_deviation = std_dev(image);
  CHECK(standard_deviation == doctest::Approx(1.118033989).epsilon(1e-9));
}

TEST_CASE("comparison of standard deviation with that of medicis") {
  t_MatrixD image(1, 4);

  image << 1, 2, 3, 4;

  auto standard_deviation = std_dev(image);
  double standard_deviation_medicis = std_dev_medicis(image);
  CHECK(standard_deviation == standard_deviation_medicis);
}