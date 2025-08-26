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
#include "global_conftest.hpp"
#include "zncc.hpp"

TYPE_TO_STRING_AS("Float", P2d::Matrixf);
TYPE_TO_STRING_AS("Double", P2d::MatrixD);

TEST_CASE("shift_image") {
  SUBCASE("shift of 0 gives two identical images gives 1") {
    P2d::Matrixf image(3, 3);
    image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    P2d::Matrixf shifted_image = shift_image(image, 0, 0);

    check_inside_eigen_element<P2d::Matrixf>(image, shifted_image);
  }
  SUBCASE("shift of 1 in row") {
    P2d::Matrixf image(3, 3);
    image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    P2d::Matrixf shifted_image = shift_image(image, 1, 0);

    P2d::Matrixf ground_truth(3, 3);
    ground_truth << 3, 4, 5, 6, 7, 8, 6, 7, 8;

    check_inside_eigen_element<P2d::Matrixf>(ground_truth, shifted_image);
  }
  SUBCASE("shift of -2 in col") {
    P2d::Matrixf image(4, 4);
    image << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16;

    P2d::Matrixf shifted_image = shift_image(image, 0, -2);

    P2d::Matrixf ground_truth(4, 4);
    ground_truth << 0, 0, 0, 1, 4, 4, 4, 5, 8, 8, 8, 9, 12, 12, 12, 13;

    check_inside_eigen_element<P2d::Matrixf>(ground_truth, shifted_image);
  }
  SUBCASE("shift of 2 in row and -1 in col") {
    P2d::Matrixf image(5, 5);
    image << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 18, 20, 21, 22, 23, 24,
        25;

    P2d::Matrixf shifted_image = shift_image(image, 2, -1);

    P2d::Matrixf ground_truth(5, 5);
    ground_truth << 10, 10, 11, 12, 13, 16, 16, 17, 18, 18, 21, 21, 22, 23, 24, 21, 21, 22, 23, 24,
        21, 21, 22, 23, 24;

    check_inside_eigen_element<P2d::Matrixf>(ground_truth, shifted_image);
  }
}

TEST_CASE_TEMPLATE("compute_integral_image", Matrix, P2d::Matrixf, P2d::MatrixD) {
  SUBCASE("Nominal case") {
    P2d::Matrixf image(3, 3);
    image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    // The integral images must be one row and one column larger than the input image.
    Matrix integral(4, 4);
    Matrix integral_sq(4, 4);

    compute_integral_image(image, integral, integral_sq);

    Matrix gt_integral(4, 4);
    Matrix gt_integral_sq(4, 4);
    gt_integral << 0, 0, 0, 0, 0, 0, 1, 3, 0, 3, 8, 15, 0, 9, 21, 36;
    gt_integral_sq << 0, 0, 0, 0, 0, 0, 1, 5, 0, 9, 26, 55, 0, 45, 111, 204;

    check_inside_eigen_element<Matrix>(gt_integral, integral);
    check_inside_eigen_element<Matrix>(gt_integral_sq, integral_sq);
  }
}

TEST_CASE_TEMPLATE("compute_right_integrals", Matrix, P2d::Matrixf, P2d::MatrixD) {
  SUBCASE("Nominal case") {
    P2d::Matrixf left_image(3, 3);
    P2d::Matrixf right_image(3, 3);

    left_image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    right_image << 1, 1, 2, 3, 4, 2, 2, 3, 4;

    // The integral images must be one row and one column larger than the input image.
    Matrix integral(4, 4);
    Matrix integral_sq(4, 4);
    Matrix integral_cross(4, 4);

    compute_right_integrals(left_image, right_image, integral, integral_sq, integral_cross);

    Matrix gt_integral(4, 4);
    Matrix gt_integral_sq(4, 4);
    Matrix gt_integral_cross(4, 4);

    gt_integral << 0, 0, 0, 0, 0, 1, 2, 4, 0, 4, 9, 13, 0, 6, 14, 22;

    gt_integral_sq << 0, 0, 0, 0, 0, 1, 2, 6, 0, 10, 27, 35, 0, 14, 40, 64;

    gt_integral_cross << 0, 0, 0, 0, 0, 0, 1, 5, 0, 9, 26, 40, 0, 21, 59, 105;

    check_inside_eigen_element<Matrix>(gt_integral, integral);
    check_inside_eigen_element<Matrix>(gt_integral_sq, integral_sq);
    check_inside_eigen_element<Matrix>(gt_integral_cross, integral_cross);
  }
}

TEST_CASE("sum_window") {
  SUBCASE("2x2 window") {
    // This integral corresponds to the image :
    // 0, 1, 2,
    // 3, 4, 5,
    // 6, 7, 8;
    P2d::MatrixD integral(4, 4);
    integral << 0, 0, 0, 0, 0, 0, 1, 3, 0, 3, 8, 15, 0, 9, 21, 36;

    double sum_win = sum_window(integral, 0, 1, 1, 2);
    double gt_sum_win = 12;
    CHECK(sum_win == gt_sum_win);
  }
  SUBCASE("1x1 window") {
    // This integral corresponds to the image :
    // 0, 1, 2,
    // 3, 4, 5,
    // 6, 7, 8;
    P2d::MatrixD integral(4, 4);
    integral << 0, 0, 0, 0, 0, 0, 1, 3, 0, 3, 8, 15, 0, 9, 21, 36;

    double sum_win = sum_window(integral, 1, 1, 1, 1);
    double gt_sum_win = 4;
    CHECK(sum_win == gt_sum_win);
  }
  SUBCASE("3x3 window") {
    // This integral corresponds to the image :
    // 1, 1, 2,
    //  3, 4, 2,
    //  2, 3, 4;
    P2d::Matrixf integral(4, 4);
    integral << 0, 0, 0, 0, 0, 1, 2, 4, 0, 4, 9, 13, 0, 6, 14, 22;

    float sum_win = sum_window(integral, 0, 0, 2, 2);
    float gt_sum_win = 22;
    CHECK(sum_win == gt_sum_win);
  }
}

TEST_CASE_TEMPLATE("ZNCC", Matrix, P2d::Matrixf, P2d::MatrixD) {
  SUBCASE("ZNCC with two identical images gives 1") {
    P2d::Matrixf image(3, 3);

    image << 0, 1, 20, 30, 0, 0, 0, 0, 0;

    // The integral images must be one row and one column larger than the input image.
    Matrix integral_left(4, 4), integral_left_sq(4, 4);
    Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

    compute_integral_image(image, integral_left, integral_left_sq);
    compute_right_integrals(image, image, integral_right, integral_right_sq, integral_cross);

    // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
    // and the window size is 3.
    auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                               integral_cross, 0, 0, 2, 2, 3);
    CHECK(zncc == doctest::Approx(1));
  }

  SUBCASE("Null standard deviation gives 0") {
    SUBCASE("Left standard deviation is null") {
      P2d::Matrixf left_image = P2d::Matrixf::Zero(3, 3);

      P2d::Matrixf right_image(3, 3);
      right_image << 0, 1, 20, 30, 0, 0, 0, 0, 0;

      // The integral images must be one row and one column larger than the input image.
      Matrix integral_left(4, 4), integral_left_sq(4, 4);
      Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

      compute_integral_image(left_image, integral_left, integral_left_sq);
      compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                              integral_cross);

      // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
      // and the window size is 3.
      auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                                 integral_cross, 0, 0, 2, 2, 3);
      CHECK(zncc == 0);
    }

    SUBCASE("Right standard deviation is null") {
      P2d::Matrixf left_image(3, 3);
      left_image << 0, 1, 20, 30, 0, 0, 0, 0, 0;

      P2d::Matrixf right_image = P2d::Matrixf::Constant(3, 3, 99.f);

      // The integral images must be one row and one column larger than the input image.
      Matrix integral_left(4, 4), integral_left_sq(4, 4);
      Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

      compute_integral_image(left_image, integral_left, integral_left_sq);
      compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                              integral_cross);

      // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
      // and the window size is 3.
      auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                                 integral_cross, 0, 0, 2, 2, 3);
      CHECK(zncc == 0);
    }
  }

  SUBCASE("Extra-small standard deviation gives 0") {
    SUBCASE("Left standard deviation is small") {
      P2d::Matrixf left_image(3, 3);
      left_image << 0, 0, 0, 0, 1e-8, 0, 0, 0, 0;

      P2d::Matrixf right_image(3, 3);
      right_image << 0, 0, 0, 450, 9, 239, 0, 0, 0;

      // The integral images must be one row and one column larger than the input image.
      Matrix integral_left(4, 4), integral_left_sq(4, 4);
      Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

      compute_integral_image(left_image, integral_left, integral_left_sq);
      compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                              integral_cross);

      // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
      // and the window size is 3.
      auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                                 integral_cross, 0, 0, 2, 2, 3);
      CHECK(zncc == 0);
    }

    SUBCASE("Right standard deviation is small") {
      P2d::Matrixf left_image(3, 3);
      left_image << 0, 0, 0, 0, 18, 999, 0, 0, 0;

      P2d::Matrixf right_image(3, 3);
      right_image << 0, 0, 0, 0, 1e-8, 0, 0, 0, 0;

      // The integral images must be one row and one column larger than the input image.
      Matrix integral_left(4, 4), integral_left_sq(4, 4);
      Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

      compute_integral_image(left_image, integral_left, integral_left_sq);
      compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                              integral_cross);

      // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
      // and the window size is 3.
      auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                                 integral_cross, 0, 0, 2, 2, 3);
      CHECK(zncc == 0);
    }
  }

  SUBCASE("ZNCC with two different images") {
    P2d::Matrixf left_image(3, 3);
    left_image << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    P2d::Matrixf right_image(3, 3);
    right_image << 0, 0, 1, 0, 3, 4, 0, 6, 7;

    // The integral images must be one row and one column larger than the input image.
    Matrix integral_left(4, 4), integral_left_sq(4, 4);
    Matrix integral_right(4, 4), integral_right_sq(4, 4), integral_cross(4, 4);

    compute_integral_image(left_image, integral_left, integral_left_sq);
    compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                            integral_cross);

    // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=2, right_col=2
    // and the window size is 3.
    auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                               integral_cross, 0, 0, 2, 2, 3);
    CHECK(zncc == doctest::Approx(0.78699100).epsilon(1e-7));
  }

  SUBCASE("ZNCC with two anticorrelated images") {
    P2d::Matrixf left_image(5, 5);
    left_image << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4;

    P2d::Matrixf right_image(5, 5);
    right_image << 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    // The integral images must be one row and one column larger than the input image.
    Matrix integral_left(6, 6), integral_left_sq(6, 6);
    Matrix integral_right(6, 6), integral_right_sq(6, 6), integral_cross(6, 6);

    compute_integral_image(left_image, integral_left, integral_left_sq);
    compute_right_integrals(left_image, right_image, integral_right, integral_right_sq,
                            integral_cross);

    // We compute zncc for the entire image so top_row=0, left_col=0, bottom_row=4, right_col=4
    // and the window size is 5.
    auto zncc = calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                               integral_cross, 0, 0, 4, 4, 5);

    CHECK(zncc == doctest::Approx(-1));
  }
}