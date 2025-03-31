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
This module contains tests associated to the Bicubic filter class for cpp.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "bicubic.hpp"
#include "doctest.h"

TEST_CASE("Test constructor") {
  Bicubic b;

  CHECK(b.get_alpha() == -0.5);
  CHECK(b.get_size() == 4);
  CHECK(b.get_margins() == Margins(1, 1, 2, 2));
}

TEST_CASE("Test result of get_coeffs computation") {
  Bicubic b;
  P2d::VectorD expected_vec(4);

  SUBCASE("fractional_shift = 0.0") {
    expected_vec << 0, 1, 0, 0;
    CHECK(b.get_coeffs(0.0) == expected_vec);
  }

  SUBCASE("fractional_shift = 0.5") {
    expected_vec << -0.0625, 0.5625, 0.5625, -0.0625;
    CHECK(b.get_coeffs(0.5) == expected_vec);
  }

  SUBCASE("fractional_shift=0.25") {
    expected_vec << -0.0703125, 0.8671875, 0.2265625, -0.0234375;
    CHECK(b.get_coeffs(0.25) == expected_vec);
  }
}