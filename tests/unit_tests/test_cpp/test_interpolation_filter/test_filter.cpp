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
This module contains tests associated to the filter class for cpp.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
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

}  // namespace abstractfilter
