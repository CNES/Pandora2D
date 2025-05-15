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
This module contains tests associated to dichotomy.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "bicubic.hpp"
#include "dichotomy.hpp"

namespace py = pybind11;

TEST_CASE("search_new_best_point") {
  double precision = 0.5;
  double subpixel = 1;
  double pos_row_disp = 2;
  double pos_col_disp = 2;
  double score = 1;
  Bicubic b;

  SUBCASE("Initial is best") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2);
    CHECK(pos_col_disp == 2);
    CHECK(score == 1);
  }

  SUBCASE("Bottom left is best") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 20., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 1.5);
    CHECK(score == 6.64453125);
  }

  SUBCASE("Bottom left is best at 0.25 precision") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 20., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2.5;
    pos_col_disp = 1.5;
    precision = 0.25;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.75);
    CHECK(pos_col_disp == 1.25);
    CHECK(score == 15.09161376953125);
  }

  SUBCASE("NaN in kernel gives initial position") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., std::numeric_limits<double>::quiet_NaN(), 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 20., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.5;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2);
    CHECK(pos_col_disp == 2);
    CHECK(score == 1);
  }

  SUBCASE("Bottom right is best") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 0., 20., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.5;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 2.5);
    CHECK(score == 6.64453125);
  }

  SUBCASE("NaN outside of kernel has no effect") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << std::numeric_limits<double>::quiet_NaN(), 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 0., 20., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.5;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 2.5);
    CHECK(score == 6.64453125);
  }

  SUBCASE("Bottom left is best and subpix=2") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 20., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.25;
    subpixel = 2;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 1.5);
    CHECK(score == 6.64453125);
  }

  SUBCASE("Bottom right is best and subpix=2") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 0., 20., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.125;
    subpixel = 2;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.25);
    CHECK(pos_col_disp == 2.25);
    CHECK(score == 1.77862548828125);
  }

  SUBCASE("Bottom left is best and subpix=4") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 20., 0., 0., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.125;
    subpixel = 4;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 1.5);
    CHECK(score == 6.64453125);
  }

  SUBCASE("Bottom right is best and subpix=4") {
    P2d::MatrixD cost_surface_data(5, 5);
    // clang-format off
    cost_surface_data << 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 0., 20., 0.,
                         0., 0., 0., 0., 0.;
    // clang-format on
    pos_row_disp = 2;
    pos_col_disp = 2;
    precision = 0.0625;
    subpixel = 4;
    search_new_best_point(cost_surface_data, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          b, "max");
    CHECK(pos_row_disp == 2.25);
    CHECK(pos_col_disp == 2.25);
    CHECK(score == 1.77862548828125);
  }
}