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
This module contains tests associated to mutual information computation.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "zncc.hpp"

TYPE_TO_STRING_AS("Float", P2d::Matrixf);
TYPE_TO_STRING_AS("Double", P2d::MatrixD);

TEST_CASE_TEMPLATE("ZNCC", Matrix, P2d::Matrixf, P2d::MatrixD) {
  SUBCASE("ZNCC with two identical images gives 1") {
    Matrix image(2, 4);

    image << 0, 1, 20, 30, 0, 0, 0, 0;

    auto zncc = calculate_zncc(image, image);
    CHECK(zncc == doctest::Approx(1));
  }

  SUBCASE("Null standard deviation gives 0") {
    SUBCASE("Left standard deviation is null") {
      Matrix left_image = Matrix::Zero(2, 4);

      Matrix right_image(2, 4);
      right_image << 0, 1, 20, 30, 0, 0, 0, 0;

      auto zncc = calculate_zncc(left_image, right_image);
      CHECK(zncc == 0);
    }

    SUBCASE("Right standard deviation is null") {
      Matrix left_image(2, 4);
      left_image << 0, 1, 20, 30, 0, 0, 0, 0;

      Matrix right_image = Matrix::Constant(2, 4, 99);

      auto zncc = calculate_zncc(left_image, right_image);
      CHECK(zncc == 0);
    }
  }

  SUBCASE("Extra-small standard deviation gives 0") {
    SUBCASE("Left standard deviation is small") {
      Matrix left_image(2, 2);
      left_image << 0, 18, 999, 0;

      Matrix right_image(2, 2);
      right_image << 0, 1e-8, 0, 0;

      auto zncc = calculate_zncc(left_image, right_image);
      CHECK(zncc == 0);
    }

    SUBCASE("Right standard deviation is small") {
      Matrix left_image(2, 2);
      left_image << 0, 1e-8, 0, 0;

      Matrix right_image(2, 2);
      right_image << 450, 9, 239, 0;

      auto zncc = calculate_zncc(left_image, right_image);
      CHECK(zncc == 0);
    }
  }

  SUBCASE("ZNCC with two different images") {
    Matrix left_image(3, 3);
    left_image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    Matrix right_image(3, 3);
    right_image << 0, 0, 1, 0, 3, 4, 0, 6, 7;

    auto zncc = calculate_zncc(left_image, right_image);
    CHECK(zncc == doctest::Approx(0.78699100).epsilon(1e-7));
  }

  SUBCASE("ZNCC with two anticorrelated images") {
    Matrix left_image(4, 4);
    left_image << 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4;

    Matrix right_image(4, 4);
    right_image << 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1;
    auto zncc = calculate_zncc(left_image, right_image);
    CHECK(zncc == doctest::Approx(-1));
  }
}