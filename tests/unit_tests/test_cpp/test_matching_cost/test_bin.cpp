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
This module contains tests associated to bin for histogram.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "bin.hpp"

TYPE_TO_STRING_AS("Float", P2d::Matrixf);
TYPE_TO_STRING_AS("Double", P2d::MatrixD);

TEST_CASE_TEMPLATE("get_bins_width_scott method", MatrixType, P2d::Matrixf, P2d::MatrixD) {
  MatrixType image(2, 4);

  SUBCASE("same cols") {
    image << 0, 1, 2, 3, 0, 1, 2, 3;

    auto bin_width = get_bins_width_scott(image);
    CHECK(bin_width == doctest::Approx(1.9515283).epsilon(1e-7));
  }

  SUBCASE("same rows") {
    image << 0, 0, 0, 0, 2, 2, 2, 2;

    auto bin_width = get_bins_width_scott(image);
    CHECK(bin_width == doctest::Approx(1.7455).epsilon(1e-7));
  }

  SUBCASE("null matrix") {
    image << 0, 0, 0, 0, 0, 0, 0, 0;

    auto bin_width = get_bins_width_scott(image);
    CHECK(bin_width == 1);
  }
}