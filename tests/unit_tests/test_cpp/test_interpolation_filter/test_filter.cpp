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
#include <functional>
#include "bicubic.hpp"

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
    expected_positions << 0.31640625, 0.5625, 0.31640625, 0.5625, 1.0, 0.5625, 0.31640625, 0.5625,
        0.31640625;
    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Interpolation around the center and precision=0.25") {
    image << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    expected_positions << 0.75201416015625, 0.8671875, 0.75201416015625, 0.8671875, 1.0, 0.8671875,
        0.75201416015625, 0.8671875, 0.75201416015625;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Best candidate at the center and subpixel shift close to 1") {
    image << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    col_positions << 1.99999999, 1.99999999, 1.99999999, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.99999999, 2.0, 2.25, 1.99999999, 2.0, 2.25, 1.99999999, 2.0, 2.25;
    expected_positions << 0.9999809489561501, 0.9999904744327068, 0.867179239547113,
        0.9999904744327068, 1.0, 0.8671875, 0.867179239547113, 0.8671875, 0.75201416015625;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical rows and precision=0.5") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    expected_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical rows and precision=0.25") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    expected_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical columns and precision=0.5") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    expected_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("Identical columns and precision=0.25") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25;
    row_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;
    expected_positions << 1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("4/5 identical rows and precision=0.5") {
    image << 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 10, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    expected_positions << 4.03125, 6.0, 4.03125, 6.5, 10.0, 6.5, 5.03125, 7.0, 5.03125;

    interpolated_positions = filter.interpolate(image, col_positions, row_positions);
    CHECK(interpolated_positions == expected_positions);
  }

  SUBCASE("4/5 identical columns and precision=0.5") {
    image << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 10, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4;

    col_positions << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    row_positions << 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5;
    expected_positions << 4.03125, 6.5, 5.03125, 6.0, 10.0, 7.0, 4.03125, 6.5, 5.03125;

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

TEST_CASE("AbstractFilter interpolate_window") {
  Bicubic filter;

  /*
  Bicubic filter dimensions (not repeated here): size 4x4 (AbstractFilter::m_size in
  interpolation_filter.hpp), margins {1, 1, 2, 2} set in bicubic.cpp (Margins: left, up, right,
  down). Asserted in test_bicubic.cpp. interpolate_window reads them via filter.get_size() and
  filter.get_margins(). Test images are 9x9 (make_image). For a 3x3 window, valid centers lie in
  [2, 6) on each axis; see border SUBCASEs below.
  */

  // Bicubic interpolation is exact on an affine image, then expected windows are known analytically
  const P2d::MatrixD col_ramp = make_image([](int, int col) { return col; });
  const P2d::MatrixD row_ramp = make_image([](int row, int) { return row; });
  const P2d::MatrixD sum_ramp = make_image([](int row, int col) { return row + col; });
  const P2d::MatrixD dirac =
      make_image([](int row, int col) { return row == 4 && col == 4 ? 1.0 : 0.0; });

  P2d::Matrixf expected_window(3, 3);

  SUBCASE("Integer center") {
    auto window = filter.interpolate_window(col_ramp, 3, 4.0, 4.0);
    REQUIRE(window.has_value());
    CHECK(*window == col_ramp.block(3, 3, 3, 3).cast<float>());
  }

  SUBCASE("Fractional center in columns and precision=0.5") {
    auto window = filter.interpolate_window(col_ramp, 3, 4.0, 3.5);
    REQUIRE(window.has_value());
    expected_window << 2.5, 3.5, 4.5, 2.5, 3.5, 4.5, 2.5, 3.5, 4.5;
    CHECK(*window == expected_window);
  }

  SUBCASE("Fractional center in columns and precision=0.25") {
    auto window = filter.interpolate_window(col_ramp, 3, 4.0, 3.25);
    REQUIRE(window.has_value());
    expected_window << 2.25, 3.25, 4.25, 2.25, 3.25, 4.25, 2.25, 3.25, 4.25;
    CHECK(*window == expected_window);
  }

  SUBCASE("Fractional center in columns and precision=0.75") {
    auto window = filter.interpolate_window(col_ramp, 3, 4.0, 3.75);
    REQUIRE(window.has_value());
    expected_window << 2.75, 3.75, 4.75, 2.75, 3.75, 4.75, 2.75, 3.75, 4.75;
    CHECK(*window == expected_window);
  }

  SUBCASE("Fractional center in rows and precision=0.5") {
    auto window = filter.interpolate_window(row_ramp, 3, 3.5, 4.0);
    REQUIRE(window.has_value());
    expected_window << 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 4.5, 4.5, 4.5;
    CHECK(*window == expected_window);
  }

  SUBCASE("Fractional center in rows and precision=0.25") {
    auto window = filter.interpolate_window(row_ramp, 3, 3.25, 4.0);
    REQUIRE(window.has_value());
    expected_window << 2.25, 2.25, 2.25, 3.25, 3.25, 3.25, 4.25, 4.25, 4.25;
    CHECK(*window == expected_window);
  }

  SUBCASE("Fractional center in rows and columns") {
    auto window = filter.interpolate_window(sum_ramp, 3, 3.5, 3.25);
    REQUIRE(window.has_value());
    expected_window << 4.75, 5.75, 6.75, 5.75, 6.75, 7.75, 6.75, 7.75, 8.75;
    CHECK(*window == expected_window);
  }

  SUBCASE("Dirac image and precision=0.5") {
    auto window = filter.interpolate_window(dirac, 3, 3.5, 3.5);
    REQUIRE(window.has_value());
    expected_window << 0.00390625, -0.03515625, -0.03515625, -0.03515625, 0.31640625, 0.31640625,
        -0.03515625, 0.31640625, 0.31640625;
    CHECK(*window == expected_window);
  }

  SUBCASE("Dirac image and precision=0.25") {
    auto window = filter.interpolate_window(dirac, 3, 3.25, 3.25);
    REQUIRE(window.has_value());
    expected_window << 0.00054931640625, -0.00531005859375, -0.02032470703125, -0.00531005859375,
        0.05133056640625, 0.19647216796875, -0.02032470703125, 0.19647216796875, 0.75201416015625;
    CHECK(*window == expected_window);
  }

  SUBCASE("Subpixel shift close to 1") {
    // The fractional shift is replaced by MAX_FRACTIONAL_VALUE to avoid rounding
    auto window = filter.interpolate_window(col_ramp, 3, 4.0, 3.99999999);
    REQUIRE(window.has_value());
    expected_window << 2.998046875, 3.998046875, 4.998046875, 2.998046875, 3.998046875, 4.998046875,
        2.998046875, 3.998046875, 4.998046875;
    CHECK(*window == expected_window);
  }

  SUBCASE("Window size of 1") {
    auto window = filter.interpolate_window(col_ramp, 1, 4.0, 3.5);
    REQUIRE(window.has_value());
    P2d::Matrixf expected_single_pixel(1, 1);
    expected_single_pixel << 3.5;
    CHECK(*window == expected_single_pixel);
  }

  SUBCASE("Window size of 5") {
    auto window = filter.interpolate_window(col_ramp, 5, 4.0, 4.5);
    REQUIRE(window.has_value());
    P2d::Matrixf expected_large_window(5, 5);
    expected_large_window << 2.5, 3.5, 4.5, 5.5, 6.5, 2.5, 3.5, 4.5, 5.5, 6.5, 2.5, 3.5, 4.5, 5.5,
        6.5, 2.5, 3.5, 4.5, 5.5, 6.5, 2.5, 3.5, 4.5, 5.5, 6.5;
    CHECK(*window == expected_large_window);
  }

  SUBCASE("Centers on the limits of the interpolable area") {
    CHECK(filter.interpolate_window(col_ramp, 3, 2.0, 2.0).has_value());
    CHECK(filter.interpolate_window(col_ramp, 3, 5.0, 5.0).has_value());
  }

  SUBCASE("Window out of the image") {
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 0.0, 4.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 8.0, 4.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 4.0, 0.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 4.0, 8.0).has_value());
  }

  SUBCASE("Window in the image but filter margins out of the image") {
    // The 3x3 windows fit in the 9x9 image, but not the resampling areas of the filter
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 1.0, 4.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 7.0, 4.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 4.0, 1.0).has_value());
    CHECK_FALSE(filter.interpolate_window(col_ramp, 3, 4.0, 7.0).has_value());
  }
}

}  // namespace abstractfilter
