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
This module contains tests associated to the filter class for cpp.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <cmath>
#include <functional>
#include <limits>
#include "bicubic.hpp"
#include "global_conftest.hpp"

namespace abstractfilter {

TEST_SUITE("AbstractFilter apply") {
  Bicubic filter;

  P2d::MatrixD resampling_area(4, 4);
  P2d::VectorD row_coeff(4);
  P2d::VectorD col_coeff(4);

  TEST_CASE("With identical rows in resampling area") {
    resampling_area << 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3;

    SUBCASE("0.5 in columns and in rows") {
      row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      double expected_result = 1.5;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }

    SUBCASE("0.5 in columns") {
      row_coeff << 0.0, 1.0, 0.0, 0.0;
      col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      double expected_result = 1.5;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }

    SUBCASE("0.5 in rows") {
      row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      col_coeff << 0.0, 1.0, 0.0, 0.0;
      double expected_result = 1.0;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }

    SUBCASE("0.25 in columns") {
      row_coeff << 0.0, 1.0, 0.0, 0.0;
      col_coeff << -0.0703125, 0.8671875, 0.2265625, -0.0234375;
      double expected_result = 1.25;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }

    SUBCASE("0.25 in rows") {
      row_coeff << -0.0703125, 0.8671875, 0.2265625, -0.0234375;
      col_coeff << 0.0, 1.0, 0.0, 0.0;
      double expected_result = 1.0;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }
  }

  TEST_CASE("with identical columns in resampling area") {
    resampling_area << 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3;

    SUBCASE("0.5 in columns and in rows") {
      row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      double expected_result = 1.5;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }
    SUBCASE("0.5 in columns") {
      row_coeff << 0.0, 1.0, 0.0, 0.0;
      col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      double expected_result = 1.0;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }
    SUBCASE("0.5 in rows") {
      row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      col_coeff << 0.0, 1.0, 0.0, 0.0;
      double expected_result = 1.5;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }
  }

  TEST_CASE("with 3/4 identical rows in resampling area") {
    resampling_area << 0, 1, 2, 3, 0, 1, 4, 3, 0, 1, 2, 3, 0, 1, 2, 3;

    SUBCASE("0.5 in columns") {
      row_coeff << 0.0, 1.0, 0.0, 0.0;
      col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      double expected_result = 2.625;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }

    SUBCASE("0.5 in rows") {
      row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
      col_coeff << 0.0, 1.0, 0.0, 0.0;
      double expected_result = 1.0;
      double result = filter.apply(resampling_area, row_coeff, col_coeff);
      CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
    }
  }
}

TEST_CASE("AbstractFilter interpolate") {
  Bicubic filter;
  P2d::MatrixD image(5, 5);
  P2d::VectorD col_positions(9);
  P2d::VectorD row_positions(9);
  P2d::VectorD expected_positions(9);
  P2d::VectorD interpolated_positions;

  SUBCASE("Interpolation around the center and precision=0.5") {
    image << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    // clang-format off
    expected_positions << 0.31640625, 0.5625, 0.31640625,
                         0.5625,      1.0,    0.5625,
                         0.31640625,  0.5625, 0.31640625;
    // clang-format on
    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Interpolation around the center and precision=0.25") {
    image << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    // clang-format off
    expected_positions << 0.75201416015625, 0.8671875, 0.75201416015625,
                         0.8671875,         1.0,       0.8671875,
                         0.75201416015625,  0.8671875, 0.75201416015625;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Best candidate at the center and subpixel shift close to 1") {
    image << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    col_positions << 1.99999999, 1.99999999, 1.99999999, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.99999999, 2.0, 2.25, 1.99999999, 2.0, 2.25, 1.99999999, 2.0, 2.25;
    // clang-format off
    expected_positions << 0.9999809489561501, 0.9999904744327068, 0.867179239547113,
                         0.9999904744327068,  1.0,                0.8671875,
                         0.867179239547113,   0.8671875,          0.75201416015625;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical rows and precision=0.5") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    // clang-format off
    expected_positions << 1.5, 1.5, 1.5,
                         2.0, 2.0, 2.0,
                         2.5, 2.5, 2.5;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical rows and precision=0.25") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    // clang-format off
    expected_positions << 1.75, 1.75, 1.75,
                         2.0,  2.0,  2.0,
                         2.25, 2.25, 2.25;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical columns and precision=0.5") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    // clang-format off
    expected_positions << 1.5, 2.0, 2.5,
                         1.5, 2.0, 2.5,
                         1.5, 2.0, 2.5;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical columns and precision=0.25") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    // clang-format off
    expected_positions << 1.75, 2.0, 2.25,
                         1.75, 2.0, 2.25,
                         1.75, 2.0, 2.25;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("4/5 identical rows and precision=0.5") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 10, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    // clang-format off
    expected_positions << 4.03125, 6.0,  4.03125,
                         6.5,     10.0,  6.5,
                         5.03125,  7.0,  5.03125;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("4/5 identical columns and precision=0.5") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 10, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    // clang-format off
    expected_positions << 4.03125,  6.5, 5.03125,
                         6.0,      10.0, 7.0,
                         4.03125,   6.5, 5.03125;
    // clang-format on

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }
}

/**
 * @brief Build a 9x9 image whose values are given by a function of the row and col indexes
 *
 * @param value function of the row and col indexes
 * @return P2d::MatrixD
 */
P2d::MatrixD make_image(const std::function<double(int, int)>& value) {
  P2d::MatrixD image(9, 9);
  for (int row = 0; row < image.rows(); ++row) {
    for (int col = 0; col < image.cols(); ++col) {
      image(row, col) = value(row, col);
    }
  }
  return image;
}

TEST_CASE("AbstractFilter interpolate_candidate_windows") {
  Bicubic filter;

  /*
  Bicubic filter dimensions (not repeated here): size 4x4, margins {1, 1, 2, 2}. Test images are
  9x9 (make_image). interpolate_candidate_windows returns a compact grid of the 8 neighbour windows
  around origin.
  */

  // Bicubic interpolation is exact on an affine image, then expected windows are known analytically
  const P2d::MatrixD col_ramp = make_image([](int, int col) { return col; });
  const P2d::MatrixD sum_ramp = make_image([](int row, int col) { return row + col; });
  const P2d::MatrixD dirac =
      make_image([](int row, int col) { return row == 4 && col == 4 ? 1.0 : 0.0; });

  // Mask with only valid pixels (1 = valid), same shape as the 9x9 test images
  const P2d::MatrixUI right_valid_mask = P2d::MatrixUI::Ones(9, 9);

  const float nan = std::numeric_limits<float>::quiet_NaN();

  // col_ramp is exact under bicubic: grid(i, j) equals the column sampling position, except the
  // (0, 0) window which stays NaN.
  SUBCASE("Full compact grid at precision=0.5") {
    auto grid = filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 4.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected(7, 7);
    // clang-format off
    expected << 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                2.5, nan, 3.5, nan, 4.5, nan, 5.5,
                2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                2.5, nan, 3.5, nan, 4.5, nan, 5.5,
                2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                2.5, nan, 3.5, nan, 4.5, nan, 5.5,
                2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5;
    // clang-format on
    CHECK(matrixf_equal_with_nan(*grid, expected));
  }

  SUBCASE("Full compact grid at precision=0.25") {
    auto grid = filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 4.0, 0.25, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected(9, 9);
    // Families {-1, 0, +1} are concatenated, so the 9x9 is nine 3x3 neighbour blocks (center NaN).
    // clang-format off
    expected << 2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, nan,  nan,  nan,  3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, nan,  nan,  nan,  3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, nan,  nan,  nan,  3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25,
                2.75, 3.75, 4.75, 3.00, 4.00, 5.00, 3.25, 4.25, 5.25;
    // clang-format on
    CHECK(matrixf_equal_with_nan(*grid, expected));
  }

  SUBCASE("Distinct row and column origins") {
    auto grid = filter.interpolate_candidate_windows(sum_ramp, 3, 4.0, 3.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected(7, 7);
    // clang-format off
    expected << 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                4.5, nan, 5.5, nan, 6.5, nan, 7.5,
                5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                5.5, nan, 6.5, nan, 7.5, nan, 8.5,
                6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                6.5, nan, 7.5, nan, 8.5, nan, 9.5,
                7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0;
    // clang-format on
    CHECK(matrixf_equal_with_nan(*grid, expected));
  }

  SUBCASE("Dirac image and precision=0.5") {
    auto grid = filter.interpolate_candidate_windows(dirac, 3, 4.0, 4.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected_top_left_window(3, 3);
    // clang-format off
    expected_top_left_window <<  0.00390625, -0.03515625, -0.03515625,
                   -0.03515625,  0.31640625,  0.31640625,
                   -0.03515625,  0.31640625,  0.31640625;
    // clang-format on
    P2d::Matrixf top_left_window(3, 3);
    top_left_window << (*grid)(0, 0), (*grid)(0, 2), (*grid)(0, 4), (*grid)(2, 0), (*grid)(2, 2),
        (*grid)(2, 4), (*grid)(4, 0), (*grid)(4, 2), (*grid)(4, 4);
    CHECK(top_left_window == expected_top_left_window);
  }

  SUBCASE("Dirac image and precision=0.25") {
    auto grid = filter.interpolate_candidate_windows(dirac, 3, 3.5, 3.5, 0.25, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected_top_left_window(3, 3);
    // clang-format off
    expected_top_left_window <<  0.00054931640625, -0.00531005859375, -0.02032470703125,
                   -0.00531005859375,  0.05133056640625,  0.19647216796875,
                   -0.02032470703125,  0.19647216796875,  0.75201416015625;
    // clang-format on
    CHECK(grid->block(0, 0, 3, 3) == expected_top_left_window);
  }

  SUBCASE("Subpixel shift close to 1") {
    // The fractional shift is replaced by MAX_FRACTIONAL_VALUE to avoid rounding
    auto grid =
        filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 4.49999999, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    CHECK((*grid)(1, 0) == 2.998046875f);
    CHECK((*grid)(1, 2) == 3.998046875f);
    CHECK((*grid)(1, 4) == 4.998046875f);
  }

  SUBCASE("Window size of 1") {
    auto grid = filter.interpolate_candidate_windows(col_ramp, 1, 4.0, 4.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected(3, 3);
    // clang-format off
    expected << 3.5, 4.0, 4.5,
                3.5, nan, 4.5,
                3.5, 4.0, 4.5;
    // clang-format on
    CHECK(matrixf_equal_with_nan(*grid, expected));
  }

  SUBCASE("Window size of 5") {
    auto grid = filter.interpolate_candidate_windows(col_ramp, 5, 4.0, 4.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    P2d::Matrixf expected(11, 11);
    for (int row = 0; row < 11; ++row) {
      for (int col = 0; col < 11; ++col) {
        // inverse_precision = 2 and first column is 1.5.
        expected(row, col) = (row % 2 == 1 && col % 2 == 1) ? nan : (1.5f + col * 0.5f);
      }
    }
    CHECK(matrixf_equal_with_nan(*grid, expected));
  }

  SUBCASE("Partial window leaves NaN on the outside neighbour") {
    auto grid = filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 1.0, 0.5, right_valid_mask);
    REQUIRE(grid.has_value());
    CHECK(std::isnan((*grid)(1, 0)));
    CHECK_FALSE(std::isnan((*grid)(1, 6)));
  }

  SUBCASE("Grid entirely out of the image") {
    CHECK_FALSE(filter.interpolate_candidate_windows(col_ramp, 3, -5.0, 4.0, 0.5, right_valid_mask)
                    .has_value());
    CHECK_FALSE(filter.interpolate_candidate_windows(col_ramp, 3, 15.0, 4.0, 0.5, right_valid_mask)
                    .has_value());
    CHECK_FALSE(filter.interpolate_candidate_windows(col_ramp, 3, 4.0, -5.0, 0.5, right_valid_mask)
                    .has_value());
    CHECK_FALSE(filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 15.0, 0.5, right_valid_mask)
                    .has_value());
  }

  SUBCASE("Candidate with an invalid interpolation support stays NaN") {
    P2d::MatrixUI mask_with_invalid_pixel = right_valid_mask;
    // Mark one pixel as invalid. An interpolation point is skipped if its
    // filter support contains this pixel.
    mask_with_invalid_pixel(1, 1) = 0;

    // Build the grid of interpolation windows around the current best position.
    // The filter support for grid(0, 0) includes pixel (1, 1), while the support
    // for grid(0, 6) does not.
    auto grid =
        filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 4.0, 0.5, mask_with_invalid_pixel);

    REQUIRE(grid.has_value());

    // Skipped points keep the NaN value used to initialize the grid.
    CHECK(std::isnan((*grid)(0, 0)));
    // This point has a fully valid interpolation support. Since col_ramp
    // stores each pixel's column as its value, interpolation at column 5.5
    // should return 5.5.
    CHECK((*grid)(0, 6) == doctest::Approx(5.5f));
  }

  SUBCASE("Multiple interpolation points are skipped for an invalid pixel") {
    P2d::MatrixUI mask_with_invalid_pixel = right_valid_mask;

    // This pixel is inside the bicubic support of several grid points.
    mask_with_invalid_pixel(3, 3) = 0;

    auto grid =
        filter.interpolate_candidate_windows(col_ramp, 3, 4.0, 4.0, 0.5, mask_with_invalid_pixel);

    REQUIRE(grid.has_value());

    // Every sample in this 5x5 region has a bicubic support containing pixel (3, 3).
    // Four points — (1, 1), (1, 3), (3, 1), and (3, 3) — also belong to the
    // initial best candidate window and are skipped independently of the mask.
    // The other samples in this region are skipped because their support contains
    // the invalid pixel.
    for (int row = 0; row <= 4; ++row) {
      for (int col = 0; col <= 4; ++col) {
        CHECK(std::isnan((*grid)(row, col)));
      }
    }

    // The support of this point does not include pixel (3, 3), so it is still interpolated.
    CHECK((*grid)(6, 6) == doctest::Approx(5.5f));
  }
}

}  // namespace abstractfilter
