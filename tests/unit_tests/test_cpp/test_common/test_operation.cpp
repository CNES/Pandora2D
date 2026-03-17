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
This module contains tests associated to the operation functions define on operation.hpp file.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "operation.hpp"


TEST_CASE("nanargmin & nanargmax") {
  SUBCASE("Positive value") {
    // Exemple d'utilisation
    P2d::VectorD data(5);
    data << 1.0, 2.0, 2.2, 3.0, 4.0;

    CHECK(nanargmin(data) == 0);
    CHECK(nanargmax(data) == 4);
  }

  SUBCASE("Positive value & Nan") {
    // Exemple d'utilisation
    P2d::VectorD data(5);
    data << 1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0, 4.0;

    CHECK(nanargmin(data) == 0);
    CHECK(nanargmax(data) == 4);
  }

  SUBCASE("Negative value") {
    // Exemple d'utilisation
    P2d::VectorD data(5);
    data << -1.0, 2.0, 2.2, 3.0, -4.0;

    CHECK(nanargmin(data) == 4);
    CHECK(nanargmax(data) == 3);
  }

  SUBCASE("Negative value & Nan") {
    // Exemple d'utilisation
    P2d::VectorD data(5);
    data << -1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0, -4.0;

    CHECK(nanargmin(data) == 4);
    CHECK(nanargmax(data) == 3);
  }

  SUBCASE("Same value") {
    // Exemple d'utilisation
    P2d::VectorD data(5);
    data << -1.0, -1.0, -1.0, -1.0, -1.0;

    CHECK(nanargmin(data) == 0);
    CHECK(nanargmax(data) == 0);
  }
}

TEST_CASE("all_same") {
  P2d::VectorD data(5);
  data << -1.0, -1.0, -1.0, -1.0, -1.0;

  SUBCASE("True") {
    CHECK(all_same(data) == true);
  }

  SUBCASE("False") {
    data[0] = 2.0;
    CHECK(all_same(data) == false);
  }
}
