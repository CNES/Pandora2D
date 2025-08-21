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
This module contains tests associated to method used during cost volumes computation.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <pybind11/embed.h>
#include "compute_cost_volumes.hpp"
#include "conftest.hpp"
#include "cost_volume.hpp"
#include "global_conftest.hpp"

template <typename window_element_type, typename window_matrix_type>
struct TypePairWindow {
  using WindowElementType = window_element_type;
  using WindowMatrixType = window_matrix_type;
};

TYPE_TO_STRING_AS("Float", TypePairWindow<float, P2d::Matrixf>);
TYPE_TO_STRING_AS("Double", TypePairWindow<double, P2d::MatrixD>);

TEST_CASE_TEMPLATE("Test get_window method",
                   T,
                   TypePairWindow<float, P2d::Matrixf>,
                   TypePairWindow<double, P2d::MatrixD>) {
  using WindowElementType = typename T::WindowElementType;
  using WindowMatrixType = typename T::WindowMatrixType;

  P2d::Matrixf img(5, 5);
  // clang-format off
  img << 1.0, 2.0, 3.0, 4.0, 5.0, 
         6.0, 7.0, 8.0, 9.0, 10.0, 
         11.0, 12.0, 13.0, 14.0, 15.0, 
         16.0, 17.0, 18.0, 19.0, 20.0, 
         21.0, 22.0, 23.0, 24.0, 25.0;
  // clang-format on

  SUBCASE("1x1 window") {
    WindowMatrixType window_gt(1, 1);
    window_gt << 8.0;
    WindowMatrixType window = get_window<WindowElementType>(img, 1, 1, 2);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
  }

  SUBCASE("3x3 window") {
    WindowMatrixType window_gt(3, 3);
    window_gt << 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0;
    WindowMatrixType window = get_window<WindowElementType>(img, 3, 2, 2);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
  }

  SUBCASE("3x3 window on the border") {
    WindowMatrixType window_gt(2, 2);
    window_gt << 1.0, 2.0, 6.0, 7.0;

    WindowMatrixType window = get_window<WindowElementType>(img, 3, 0, 0);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
  }

  SUBCASE("3x3 window on the border with negative index") {
    WindowMatrixType window_gt(2, 1);
    window_gt << 1.0, 6.0;
    WindowMatrixType window = get_window<WindowElementType>(img, 3, 0, -1);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
    ;
  }

  SUBCASE("3x3 window out of the image") {
    WindowMatrixType window_gt(0, 0);
    WindowMatrixType window = get_window<WindowElementType>(img, 3, -2, -2);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
    ;
  }

  SUBCASE("5x5 window") {
    WindowMatrixType window = get_window<WindowElementType>(img, 5, 2, 2);
    check_inside_eigen_element<WindowMatrixType>(window, img.template cast<WindowElementType>());
  }

  SUBCASE("5x5 window on the border") {
    WindowMatrixType window_gt(3, 5);
    // clang-format off
    window_gt << 11.0, 12.0, 13.0, 14.0, 15.0, 
                 16.0, 17.0, 18.0, 19.0, 20.0, 
                 21.0, 22.0, 23.0, 24.0, 25.0;
    // clang-format on
    WindowMatrixType window = get_window<WindowElementType>(img, 5, 4, 2);
    check_inside_eigen_element<WindowMatrixType>(window, window_gt);
  }
}

TEST_CASE("Test get_index_right method") {
  /*
  When subpix=1, we have a single right image
  */

  SUBCASE("Subpix=1") {
    CHECK(interpolated_right_image_index(1, 1, 2) == 0);
  }

  /*
  When subpix=2, right images are arranged in this order (fmod is the fractional part):

  fmod(d_row) | fmod(d_col)
  0           | 0
  0           | 0.5
  0.5         | 0
  0.5         | 0.5
  */

  // only the fractional part of the disparities is taken into account
  // when calculating the index of the interpolated right image
  SUBCASE("Subpix=2") {
    CHECK(interpolated_right_image_index(2, 1, 2) == 0);
    CHECK(interpolated_right_image_index(2, 1, 2.5) == 1);
    CHECK(interpolated_right_image_index(2, 1.5, 2.) == 2);
    CHECK(interpolated_right_image_index(2, 1.5, 2.5) == 3);

    CHECK(interpolated_right_image_index(2, 1., -2.5) == 1);
    CHECK(interpolated_right_image_index(2, -1.5, 2.) == 2);
    CHECK(interpolated_right_image_index(2, -1.5, -2.5) == 3);
  }

  /*
  When subpix=4, right images are arranged in this order (fmod is the fractional part):

  fmod(d_row) | fmod(d_col)
  0           | 0
  0           | 0.25
  0           | 0.5
  0           | 0.75
  0.25        | 0
  0.25        | 0.25
  0.25        | 0.5
  0.25        | 0.75
  0.5         | 0
  0.5         | 0.25
  0.5         | 0.5
  0.5         | 0.75
  0.75        | 0
  0.75        | 0.25
  0.75        | 0.5
  0.75        | 0.75
  */

  // only the fractional part of the disparities is taken into account
  // when calculating the index of the interpolated right image
  SUBCASE("Subpix=4") {
    CHECK(interpolated_right_image_index(4, 1, 2) == 0);
    CHECK(interpolated_right_image_index(4, 1, 2.25) == 1);
    CHECK(interpolated_right_image_index(4, 1, 2.5) == 2);
    CHECK(interpolated_right_image_index(4, 1, 2.75) == 3);
    CHECK(interpolated_right_image_index(4, 1, -2.25) == 3);
    CHECK(interpolated_right_image_index(4, 1, -2.75) == 1);

    CHECK(interpolated_right_image_index(4, 1.25, 2) == 4);
    CHECK(interpolated_right_image_index(4, 1.25, 2.25) == 5);
    CHECK(interpolated_right_image_index(4, 1.25, 2.5) == 6);
    CHECK(interpolated_right_image_index(4, 1.25, 2.75) == 7);
    CHECK(interpolated_right_image_index(4, 1.25, -2.25) == 7);
    CHECK(interpolated_right_image_index(4, 1.25, -2.75) == 5);

    CHECK(interpolated_right_image_index(4, 1.5, 2) == 8);
    CHECK(interpolated_right_image_index(4, 1.5, 2.25) == 9);
    CHECK(interpolated_right_image_index(4, 1.5, 2.5) == 10);
    CHECK(interpolated_right_image_index(4, 1.5, 2.75) == 11);
    CHECK(interpolated_right_image_index(4, 1.5, -2.25) == 11);
    CHECK(interpolated_right_image_index(4, 1.5, -2.75) == 9);

    CHECK(interpolated_right_image_index(4, 1.75, 2) == 12);
    CHECK(interpolated_right_image_index(4, 1.75, 2.25) == 13);
    CHECK(interpolated_right_image_index(4, 1.75, 2.5) == 14);
    CHECK(interpolated_right_image_index(4, 1.75, 2.75) == 15);
    CHECK(interpolated_right_image_index(4, 1.75, -2.25) == 15);
    CHECK(interpolated_right_image_index(4, 1.75, -2.75) == 13);
  }
}

TEST_CASE("Test has_only_non_zero_elements method") {
  P2d::MatrixUI mat(3, 3);
  ;

  SUBCASE("Test a vector with only non zero elements") {
    mat << 1, 2, 3, 4, 3, 1, 2, 1, 4;
    CHECK(all_non_zero_elements(mat) == true);
  }

  SUBCASE("Test a vector with some zero elements") {
    mat << 1, 0, 0, 4, 3, 0, 2, 1, 0;
    CHECK(all_non_zero_elements(mat) == false);
  }

  SUBCASE("Test a vector with only zero elements") {
    P2d::MatrixUI mat_zero = P2d::MatrixUI::Zero(3, 3);
    CHECK(all_non_zero_elements(mat_zero) == false);
  }
}