/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

#include <cmath>
#include <limits>
#include <string>

#include "bicubic.hpp"
#include "dichotomy.hpp"

namespace py = pybind11;

TEST_CASE("search_new_best_point") {
  double precision = 0.5;
  double subpixel = 1;
  double pos_row_disp = 2;
  double pos_col_disp = 2;
  double score = 1;
  Bicubic bicubic;

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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
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
                          bicubic, "max");
    CHECK(pos_row_disp == 2.25);
    CHECK(pos_col_disp == 2.25);
    CHECK(score == 1.77862548828125);
  }
}

/**
 * @brief Build a non affine image so that ZNCC distinguishes neighbouring subpixel windows
 */
static P2d::MatrixD make_right_image(int nb_rows, int nb_cols) {
  P2d::MatrixD image(nb_rows, nb_cols);
  for (int row = 0; row < nb_rows; ++row) {
    for (int col = 0; col < nb_cols; ++col) {
      image(row, col) = row * row + 3. * col * col;
    }
  }
  return image;
}

/**
 * @brief Build a left correlation window resampled at a subpixel centre
 *
 * Used in search_new_best_point_on_image tests as the reference (left) window. For each pixel of
 * a window_size x window_size patch, subpixel coordinates are taken on a regular grid around
 * (center_row, center_col), then filter.interpolate samples the right image at those positions.
 * The returned matrix is the same layout as a matching-cost window (row-major within the patch).
 */
static P2d::Matrixf single_window(abstractfilter::AbstractFilter& filter,
                                  const P2d::MatrixD& image,
                                  int window_size,
                                  double center_row,
                                  double center_col) {
  const int half_window = window_size / 2;
  const int nb_pixels = window_size * window_size;
  P2d::VectorD row_positions(nb_pixels);
  P2d::VectorD col_positions(nb_pixels);
  int index = 0;
  for (int row = -half_window; row <= half_window; ++row) {
    for (int col = -half_window; col <= half_window; ++col) {
      row_positions[index] = center_row + row;
      col_positions[index] = center_col + col;
      ++index;
    }
  }
  const P2d::VectorD interpolated_values =
      filter.interpolate(image, col_positions, row_positions, MAX_FRACTIONAL_VALUE);
  P2d::Matrixf window(window_size, window_size);
  for (int pixel = 0; pixel < nb_pixels; ++pixel) {
    window(pixel / window_size, pixel % window_size) =
        static_cast<float>(interpolated_values[pixel]);
  }
  return window;
}

TEST_CASE("search_new_best_point_on_image") {
  const int window_size = 5;
  const int pixel_row = 7;
  const int pixel_col = 7;
  double precision = 0.5;
  double best_row_disp = 0.;
  double best_col_disp = 0.;
  double best_score = -1.;
  Bicubic bicubic;
  P2d::MatrixD right_image = make_right_image(15, 15);

  SUBCASE("Initial is best") {
    P2d::Matrixf left_window = right_image.block(5, 5, window_size, window_size).cast<float>();
    best_score = 1.;
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    CHECK(best_row_disp == 0.);
    CHECK(best_col_disp == 0.);
    CHECK(best_score == 1.);
  }

  SUBCASE("Bottom left is best") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == -0.5);
    CHECK(best_score == doctest::Approx(1.));
  }

  SUBCASE("Bottom left is best with mutual information") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "mutual_information");
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == -0.5);
    CHECK(best_score > 0.);
  }

  SUBCASE("Bottom left is best at 0.25 precision") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.75, pixel_col + 0.25);
    best_row_disp = 0.5;
    best_col_disp = 0.5;
    precision = 0.25;
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    CHECK(best_row_disp == 0.75);
    CHECK(best_col_disp == 0.25);
    CHECK(best_score == doctest::Approx(1.));
  }

  SUBCASE("Best neighbour is selected without NaN") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col + 0.5);
    double clean_best_row_disp = 0.;
    double clean_best_col_disp = 0.;
    double clean_best_score = 0.5;
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, clean_best_row_disp, clean_best_col_disp,
                                   clean_best_score, bicubic, "zncc-optim-2");
    // The best neighbour (0.5, 0.5) should be selected as it has the best score
    CHECK((clean_best_row_disp == 0.5 && clean_best_col_disp == 0.5));
    // The score must be valid (not NaN) when best neighbour is clean
    CHECK_FALSE(std::isnan(clean_best_score));
  }

  SUBCASE("NaN in kernel of the best neighbour is ignored") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col + 0.5);
    right_image(pixel_row + 4, pixel_col + 4) = std::numeric_limits<double>::quiet_NaN();
    best_score = 0.5;
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    // The poisoned neighbour (0.5, 0.5) must not be selected. Any other disparity is fine.
    CHECK((best_row_disp != 0.5 || best_col_disp != 0.5));
    // Ensure NaN in the interpolated window does not propagate to the final score
    CHECK_FALSE(std::isnan(best_score));
  }

  SUBCASE("Bottom right is best") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col + 0.5);
    search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == 0.5);
    CHECK(best_score == doctest::Approx(1.));
  }

  SUBCASE("Window out of the image") {
    P2d::MatrixD small_image = make_right_image(3, 3);
    P2d::Matrixf left_window = P2d::Matrixf::Ones(window_size, window_size);
    best_score = 0.5;
    search_new_best_point_on_image(left_window, small_image, precision, 1, 1, window_size,
                                   best_row_disp, best_col_disp, best_score, bicubic,
                                   "zncc-optim-2");
    CHECK(best_row_disp == 0.);
    CHECK(best_col_disp == 0.);
    CHECK(best_score == 0.5);
  }

  SUBCASE("zncc and zncc-optim-1 are aliased to zncc-optim-2") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    for (const char* method : {"zncc", "zncc-optim-1"}) {
      best_row_disp = 0.;
      best_col_disp = 0.;
      best_score = -1.;
      search_new_best_point_on_image(left_window, right_image, precision, pixel_row, pixel_col,
                                     window_size, best_row_disp, best_col_disp, best_score, bicubic,
                                     method);
      CHECK(best_row_disp == 0.5);
      CHECK(best_col_disp == -0.5);
      CHECK(best_score == doctest::Approx(1.));
    }
  }

  SUBCASE("Unknown method throws") {
    P2d::Matrixf left_window = right_image.block(5, 5, window_size, window_size).cast<float>();
    CHECK_THROWS_AS(search_new_best_point_on_image(left_window, right_image, precision, pixel_row,
                                                   pixel_col, window_size, best_row_disp,
                                                   best_col_disp, best_score, bicubic, "ssd"),
                    std::invalid_argument);
  }
}