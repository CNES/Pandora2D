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
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    best_score = 1.;
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
    CHECK(best_row_disp == 0.);
    CHECK(best_col_disp == 0.);
    CHECK(best_score == 1.);
  }

  SUBCASE("Bottom left is best") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == -0.5);
    CHECK(best_score == doctest::Approx(1.));
  }

  SUBCASE("Bottom left is best with mutual information") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    const Correlator<double> correlator =
        make_prepared_left_window(left_window, "mutual_information");
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == -0.5);
    CHECK(best_score > 0.);
  }

  SUBCASE("Bottom left is best at 0.25 precision") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.75, pixel_col + 0.25);
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    best_row_disp = 0.5;
    best_col_disp = 0.5;
    precision = 0.25;
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
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
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, clean_best_row_disp, clean_best_col_disp,
                                   clean_best_score, bicubic);
    // The best neighbour (0.5, 0.5) should be selected as it has the best score
    CHECK((clean_best_row_disp == 0.5 && clean_best_col_disp == 0.5));
    // The score must be valid (not NaN) when best neighbour is clean
    CHECK_FALSE(std::isnan(clean_best_score));
  }

  SUBCASE("NaN in kernel of the best neighbour is ignored") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col + 0.5);
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    right_image(pixel_row + 4, pixel_col + 4) = std::numeric_limits<double>::quiet_NaN();
    best_score = 0.5;
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
    CHECK((best_row_disp != 0.5 || best_col_disp != 0.5));
    // Ensure NaN in the interpolated window does not propagate to the final score
    CHECK_FALSE(std::isnan(best_score));
  }

  SUBCASE("Bottom right is best") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col + 0.5);
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                   window_size, best_row_disp, best_col_disp, best_score, bicubic);
    CHECK(best_row_disp == 0.5);
    CHECK(best_col_disp == 0.5);
    CHECK(best_score == doctest::Approx(1.));
  }

  SUBCASE("Window out of the image") {
    P2d::MatrixD small_image = make_right_image(3, 3);
    P2d::Matrixf left_window = P2d::Matrixf::Ones(window_size, window_size);
    const Correlator<double> correlator = make_prepared_left_window(left_window, "zncc-optim-2");
    best_score = 0.5;
    search_new_best_point_on_image(correlator, small_image, precision, 1, 1, window_size,
                                   best_row_disp, best_col_disp, best_score, bicubic);
    CHECK(best_row_disp == 0.);
    CHECK(best_col_disp == 0.);
    CHECK(best_score == 0.5);
  }

  SUBCASE("zncc and zncc-optim-1 are aliased to zncc-optim-2") {
    P2d::Matrixf left_window =
        single_window(bicubic, right_image, window_size, pixel_row + 0.5, pixel_col - 0.5);
    for (const char* method : {"zncc", "zncc-optim-1"}) {
      const Correlator<double> correlator = make_prepared_left_window(left_window, method);
      best_row_disp = 0.;
      best_col_disp = 0.;
      best_score = -1.;
      search_new_best_point_on_image(correlator, right_image, precision, pixel_row, pixel_col,
                                     window_size, best_row_disp, best_col_disp, best_score,
                                     bicubic);
      CHECK(best_row_disp == 0.5);
      CHECK(best_col_disp == -0.5);
      CHECK(best_score == doctest::Approx(1.));
    }
  }

  SUBCASE("Unknown method throws") {
    P2d::Matrixf left_window = right_image.block(5, 5, window_size, window_size).cast<float>();
    CHECK_THROWS_AS(
        search_new_best_point_on_image(make_prepared_left_window(left_window, "ssd"), right_image,
                                       precision, pixel_row, pixel_col, window_size, best_row_disp,
                                       best_col_disp, best_score, bicubic),
        std::invalid_argument);
  }
}

/**
 * @brief Build a left image equal to the right image, except for the window centered on the pixel,
 * which is the right image interpolated with a known shift
 */
static P2d::Matrixf make_left_image(abstractfilter::AbstractFilter& filter,
                                    const P2d::MatrixD& right_image,
                                    int window_size,
                                    int pixel_row,
                                    int pixel_col,
                                    double shift_row,
                                    double shift_col) {
  P2d::Matrixf left_image = right_image.cast<float>();
  const int half_window = window_size / 2;
  left_image.block(pixel_row - half_window, pixel_col - half_window, window_size, window_size) =
      single_window(filter, right_image, window_size, pixel_row + shift_row, pixel_col + shift_col);
  return left_image;
}

TEST_CASE("image resampling") {
  const int window_size = 5;
  const int pixel_row = 7;
  const int pixel_col = 7;
  // The first elements of the disparity range are :
  // [-1, -0.5, 0, 0.5, 1, ...]
  int subpixel = 2;
  const double disp_row_min = -1.;
  const double disp_col_min = -1.;
  // subpix = 2, so the first iteration of the dichotomy is skipped
  int nb_iterations = 2;
  int first_iteration = 2;
  double pos_row_disp = 2.;  // disparity value = 0
  double pos_col_disp = 2.;  // disparity value = 0
  double score = -1.;
  Bicubic bicubic;
  P2d::MatrixD right_image = make_right_image(15, 15);

  SUBCASE("Initial is best") {
    const P2d::Matrixf left_image = right_image.cast<float>();
    score = 1.;

    image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                     first_iteration, bicubic, "zncc-optim-2", subpixel, disp_row_min, disp_col_min,
                     pos_row_disp, pos_col_disp, score);

    // No neighbour can do better than a perfect score: nothing is updated
    CHECK(pos_row_disp == 2.);
    CHECK(pos_col_disp == 2.);
    CHECK(score == 1.);
  }

  SUBCASE("Center right is best at 0.5 precision") {
    subpixel = 1;
    first_iteration = 1;
    nb_iterations = 1;
    P2d::Matrixf left_image =
        make_left_image(bicubic, right_image, window_size, pixel_row, pixel_col, 0., -0.5);
    // start from disparities (0, 0) -> position indexes ((0 + 1) * 1, (0 + 1) * 1) = (1, 1)
    pos_row_disp = 1.;
    pos_col_disp = 1.;
    image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                     first_iteration, bicubic, "zncc-optim-2", subpixel, disp_row_min, disp_col_min,
                     pos_row_disp, pos_col_disp, score);
    // disparities (0, -0.5) -> position indexes ((0 + 1) * 1, (-0.5 + 1) * 1) = (1, 0.5)
    CHECK(pos_row_disp == 1.);
    CHECK(pos_col_disp == 0.5);
    CHECK(score == doctest::Approx(1.));
  }

  SUBCASE("Bottom left is best at 0.25 precision") {
    P2d::Matrixf left_image =
        make_left_image(bicubic, right_image, window_size, pixel_row, pixel_col, 0.25, -0.25);

    image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                     first_iteration, bicubic, "zncc-optim-2", subpixel, disp_row_min, disp_col_min,
                     pos_row_disp, pos_col_disp, score);

    // We return disparities as position indexes, not raw disparity values
    // disparities (0.25, -0.25) -> indexes ((0.25 - (-1)) * 2, (-0.25 - (-1)) * 2)
    CHECK(pos_row_disp == 2.5);
    CHECK(pos_col_disp == 1.5);
    CHECK(score == doctest::Approx(1.));
  }

  SUBCASE("Bottom right is best at 0.25 precision, from a non integer starting point") {
    P2d::Matrixf left_image =
        make_left_image(bicubic, right_image, window_size, pixel_row, pixel_col, 0.75, 0.25);
    // start from disparities (0.5, 0.5) -> position indexes (3, 3)
    pos_row_disp = 3.;
    pos_col_disp = 3.;
    image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                     first_iteration, bicubic, "zncc-optim-2", subpixel, disp_row_min, disp_col_min,
                     pos_row_disp, pos_col_disp, score);

    // We return disparities as position indexes, not raw disparity values
    // disparities (0.75, 0.25) -> position indexes (3.5, 2.5)
    CHECK(pos_row_disp == 3.5);
    CHECK(pos_col_disp == 2.5);
    CHECK(score == doctest::Approx(1.));
  }

  SUBCASE("Subpixel 4: precisions 0.5 and 0.25 are skipped, the first one is 0.125") {
    subpixel = 4;
    first_iteration = 3;
    nb_iterations = 3;
    P2d::Matrixf left_image =
        make_left_image(bicubic, right_image, window_size, pixel_row, pixel_col, 0.375, 0.125);
    // start from disparities (0.25, 0.25) -> indexes ((0.25 + 1) * 4, (0.25 + 1) * 4) = (5, 5)
    pos_row_disp = 5.;
    pos_col_disp = 5.;
    image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                     first_iteration, bicubic, "zncc-optim-2", subpixel, disp_row_min, disp_col_min,
                     pos_row_disp, pos_col_disp, score);
    // We return disparities as position indexes, not raw disparity values
    // disparities (0.375, 0.125) -> position indexes (5.5, 4.5)
    CHECK(pos_row_disp == 5.5);
    CHECK(pos_col_disp == 4.5);
    CHECK(score == doctest::Approx(1.));
  }

  SUBCASE("zncc and zncc-optim-1 give the same result as zncc-optim-2") {
    P2d::Matrixf left_image =
        make_left_image(bicubic, right_image, window_size, pixel_row, pixel_col, 0.25, -0.25);
    for (const char* method : {"zncc", "zncc-optim-1"}) {
      pos_row_disp = 2.;
      pos_col_disp = 2.;
      score = -1.;
      image_resampling(left_image, right_image, pixel_row, pixel_col, window_size, nb_iterations,
                       first_iteration, bicubic, method, subpixel, disp_row_min, disp_col_min,
                       pos_row_disp, pos_col_disp, score);
      // We return disparities as position indexes, not raw disparity values
      // disparities (0.25, -0.25) -> position indexes (2.5, 1.5)
      CHECK(pos_row_disp == 2.5);
      CHECK(pos_col_disp == 1.5);
      CHECK(score == doctest::Approx(1.));
    }
  }

  SUBCASE("Unknown method throws") {
    const P2d::Matrixf left_image = right_image.cast<float>();
    CHECK_THROWS_AS(image_resampling(left_image, right_image, pixel_row, pixel_col, window_size,
                                     nb_iterations, first_iteration, bicubic, "ssd", subpixel,
                                     disp_row_min, disp_col_min, pos_row_disp, pos_col_disp, score),
                    std::invalid_argument);
  }
}