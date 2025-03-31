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
#include "compute_cost_volumes.hpp"
#include "cost_volume.hpp"
#include "global_conftest.hpp"

constexpr double INIT_VALUE_CV = 0.0;  ///< initial value used to fill the cv

/**
 * @brief Get the cost surface of a cv point (row,col)
 *        row and col correspond to the cv index, for example the point (0,0)
 *        is the first point of the cv but not necessarily the first image point in the ROI case
 *
 * @param cost_values vector of cost values
 * @param cv_shape 4d cv shape
 * @param row cv index
 * @param col cv index
 * @return P2d::VectorD
 */
P2d::VectorD get_cost_surface(const P2d::VectorD& cost_values,
                              CostVolumeSize& cv_size,
                              int row,
                              int col) {
  int cost_surface_size = cv_size.nb_disps();

  int start_index = (row * cv_size.nb_col + col) * cost_surface_size;

  return cost_values.segment(start_index, cost_surface_size);
};

TEST_CASE("Test get_window method") {
  P2d::MatrixD img(5, 5);
  // clang-format off
  img << 1.0, 2.0, 3.0, 4.0, 5.0, 
         6.0, 7.0, 8.0, 9.0, 10.0, 
         11.0, 12.0, 13.0, 14.0, 15.0, 
         16.0, 17.0, 18.0, 19.0, 20.0, 
         21.0, 22.0, 23.0, 24.0, 25.0;
  // clang-format on

  SUBCASE("1x1 window") {
    P2d::MatrixD window_gt(1, 1);
    window_gt << 8.0;
    P2d::MatrixD window = get_window(img, 1, 1, 2);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
  }

  SUBCASE("3x3 window") {
    P2d::MatrixD window_gt(3, 3);
    window_gt << 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0;
    P2d::MatrixD window = get_window(img, 3, 2, 2);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
  }

  SUBCASE("3x3 window on the border") {
    P2d::MatrixD window_gt(2, 2);
    window_gt << 1.0, 2.0, 6.0, 7.0;

    P2d::MatrixD window = get_window(img, 3, 0, 0);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
  }

  SUBCASE("3x3 window on the border with negative index") {
    P2d::MatrixD window_gt(2, 1);
    window_gt << 1.0, 6.0;
    P2d::MatrixD window = get_window(img, 3, 0, -1);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
    ;
  }

  SUBCASE("3x3 window out of the image") {
    P2d::MatrixD window_gt(0, 0);
    P2d::MatrixD window = get_window(img, 3, -2, -2);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
    ;
  }

  SUBCASE("5x5 window") {
    P2d::MatrixD window = get_window(img, 5, 2, 2);
    check_inside_eigen_element<P2d::MatrixD>(window, img);
  }

  SUBCASE("5x5 window on the border") {
    P2d::MatrixD window_gt(3, 5);
    // clang-format off
    window_gt << 11.0, 12.0, 13.0, 14.0, 15.0, 
                 16.0, 17.0, 18.0, 19.0, 20.0, 
                 21.0, 22.0, 23.0, 24.0, 25.0;
    // clang-format on
    P2d::MatrixD window = get_window(img, 5, 4, 2);
    check_inside_eigen_element<P2d::MatrixD>(window, window_gt);
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

TEST_CASE("Test contains_element method") {
  P2d::MatrixD mat(3, 3);
  mat << 1.0, 2.0, 3.0, 4.0, 3.0, 6.0, 2.0, 1.0, 0.0;

  SUBCASE("Test if a double is in a matrix") {
    CHECK(contains_element(mat, 0.0) == true);
    CHECK(contains_element(mat, 5.0) == false);
  }

  SUBCASE("Test if a nan is in a matrix") {
    P2d::MatrixD mat_nan(3, 3);
    mat_nan << 1.0, 2.0, nan("1"), 4.0, 3.0, 6.0, 2.0, 1.0, 0.0;

    CHECK(contains_element(mat, nan("1")) == false);
    CHECK(contains_element(mat_nan, nan("1")) == true);
  }
}

TEST_CASE("Test compute_cost_volumes_cpp method") {
  // Left image
  P2d::MatrixD img_left(5, 5);
  // clang-format off
  img_left << 1.0, 2.0, 3.0, 4.0, 5.0, 
              6.0, 7.0, 8.0, 9.0, 10.0, 
              11.0, 12.0, 13.0, 14.0, 15.0, 
              16.0, 17.0, 18.0, 19.0, 20.0, 
              21.0, 22.0, 23.0, 24.0, 25.0;
  // clang-format on

  // Right image
  std::vector<P2d::MatrixD> imgs_right;

  P2d::MatrixD right_1(5, 5);
  // clang-format off
  right_1 << 1., 2., 3., 4., 2., 
             2., 2., 2., 2., 2., 
             4., 3., 2., 1., 4., 
             1., 3., 3., 3., 1., 
             1., 3., 2., 4., 4.;
  // clang-format on

  imgs_right.push_back(right_1);

  // CV size
  CostVolumeSize cv_size = CostVolumeSize(5, 5, 3, 5);

  // Initialized cv values
  P2d::VectorD cv_values = P2d::VectorD::Zero(cv_size.size());

  // Disparity ranges
  P2d::VectorD disp_range_row(3);
  disp_range_row << -1.0, 0.0, 1.0;
  P2d::VectorD disp_range_col(5);
  disp_range_col << -2.0, -1.0, 0.0, 1.0, 2.0;

  // Offset between cv and image first points
  int offset_cv_img_row = 0;
  int offset_cv_img_col = 0;

  // Window size
  int window_size = 3;
  // [step_row, step_col]
  Eigen::Vector2i step;
  step << 1, 1;

  // img no data value
  double no_data = -9999;

  SUBCASE("Cost surface of top left point") {
    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 0, 0);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       INIT_VALUE_CV, INIT_VALUE_CV, 0.31127812445913294, INIT_VALUE_CV, INIT_VALUE_CV, 
                       INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface of center point") {
    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 2, 2);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, 0.22478750958935989, 0.22478750958935989, 0.072780225783732888, INIT_VALUE_CV,
                       INIT_VALUE_CV, 0.22478750958935989, 0.10218717094933361, 0.37887883713522919, INIT_VALUE_CV, 
                       INIT_VALUE_CV, 0.0072146184745172093, 0.22478750958935989, 0.0072146184745172093, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface of center point with not centered disparities") {
    // CV size
    CostVolumeSize cv_size = CostVolumeSize(5, 5, 3, 3);

    // Initialized cv values
    P2d::VectorD cv_values = P2d::VectorD::Zero(cv_size.size());

    // Disparity ranges
    P2d::VectorD disp_range_row(3);
    disp_range_row << -2.0, -1.0, 0.0;
    P2d::VectorD disp_range_col(3);
    disp_range_col << 0.0, 1.0, 2.0;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 2, 2);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       0.22478750958935989, 0.072780225783732888, INIT_VALUE_CV,
                       0.10218717094933361, 0.37887883713522919, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface with step_row=2 and step_col=3") {
    // Smaller shape with step=[2,3]
    CostVolumeSize cv_size = CostVolumeSize(3, 2, 3, 5);
    cv_values = P2d::VectorD::Zero(cv_size.size());
    step << 2, 3;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 1, 1);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << 0.22478750958935989, 0.22478750958935989, 0.072780225783732888, INIT_VALUE_CV, INIT_VALUE_CV, 
                       0.22478750958935989, 0.10218717094933361, 0.37887883713522919, INIT_VALUE_CV, INIT_VALUE_CV, 
                       0.0072146184745172093, 0.22478750958935989, 0.0072146184745172093, INIT_VALUE_CV, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface with subpix=2") {
    // When subpix=2, we have 4 right images
    P2d::MatrixD right_2(5, 5);

    // clang-format off
    right_2 << 1.5, 2.5, 3.5, 3., 1., 
               2., 2., 2., 2., 2., 
               3.5, 2.5, 1.5, 2.5, 1., 
               2., 3., 3., 2., 1., 
               2., 2.5, 3., 4., 1.;
    // clang-format on

    P2d::MatrixD right_3(5, 5);
    // clang-format off
    right_3 << 1.5, 2., 2.5, 3., 2., 
               3., 2.5, 2., 1.5, 3., 
               2.5, 3., 2.5, 2., 2.5, 
               1., 3., 2.5, 3.5, 2.5, 
               1., 3., 2., 4., 4.;
    // clang-format on

    P2d::MatrixD right_4(5, 5);
    // clang-format off
    right_4 << 1.75, 2.25, 2.75, 2.5, 2., 
               2., 2.25, 1.75, 2.25, 2., 
               3., 2.75, 2.25, 2.25, 4., 
               2., 2.75, 3., 3.25, 1., 
               1., 3., 2., 4., 4.;
    // clang-format on

    imgs_right.push_back(right_2);
    imgs_right.push_back(right_3);
    imgs_right.push_back(right_4);

    // Biggest shape with subpix=2
    CostVolumeSize cv_size = CostVolumeSize(5, 5, 5, 9);
    cv_values = P2d::VectorD::Zero(cv_size.size());

    // Largest disparity ranges with subpix=2
    P2d::VectorD disp_range_row(5);
    disp_range_row << -1.0, -0.5, 0.0, 0.5, 1.0;
    P2d::VectorD disp_range_col(9);
    disp_range_col << -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 2, 2);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());
    // clang-format off
    cost_surface_gt <<
    // d_col    -2          -1.5         -1                               d_row
          INIT_VALUE_CV, INIT_VALUE_CV, 0.22478750958935989,              // -1
        //     -0.5                      0              0.5
        0.0072146184745174313, 0.22478750958935989, 0.32440939317155548,  // -1
        //        1                      1.5             2
        0.072780225783732888, 0.072780225783732888, INIT_VALUE_CV,        //-1


    // d_col   -2           -1.5         -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.072780225783732666,               // -0.5
        //  -0.5                          0              0.5
        0.018310781820059185, 0.091091007603791629, 0.14556045156746578,  // -0.5
        //   1                            1.5            2
        0.0072146184745172093, 0.37887883713522919, INIT_VALUE_CV,        // -0.5


    // d_col  -2            -1.5         -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.22478750958935989,                // 0
        //  -0.5                          0               0.5
        0.091091007603791851, 0.10218717094933361, 0.091091007603791851,  // 0
        //  1                             1.5              2
        0.37887883713522919, 0.018310781820059185, INIT_VALUE_CV,         // 0


    // d_col -2              -1.5        -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.10218717094933361,                // 0.5
        //  -0.5                          0               0.5
        0.091091007603791851, 0.072780225783732, 0.32440939317155526,     // 0.5
        //  1                             1.5             2
        0.10218717094933316, 0.22943684069673953, INIT_VALUE_CV,          // 0.5


    // d_col -2               -1.5        -1                              d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.0072146184745172093,              // 1
        //  -0.5                          0               0.5
        0.072780225783732666, 0.22478750958935989, 0.10218717094933316,   // 1
        //  1                             1.5             2
        0.0072146184745172093, 0.0072146184745174313, INIT_VALUE_CV;      // 1
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  };

  SUBCASE("Cost surface with subpix=2 and no data values") {
    // When subpix=2, we have 4 right images

    P2d::MatrixD right_2(5, 5);
    // clang-format off
    right_2 << 1.5, 2.5, 3.5, 3., -9999., 
               2., 2., 2., 2., -9999., 
               3.5, 2.5, 1.5, 2.5, -9999., 
               2., 3., 3., 2., -9999., 
               2., 2.5, 3., 4., -9999.;
    // clang-format on

    P2d::MatrixD right_3(5, 5);
    // clang-format off
    right_3 << 1.5, 2., 2.5, 3., 2., 
               3., 2.5, 2., 1.5, 3., 
               2.5, 3., 2.5, 2., 2.5, 
               1., 3., 2.5, 3.5, 2.5, 
               -9999., -9999., -9999., -9999., -9999.;
    // clang-format on

    P2d::MatrixD right_4(5, 5);
    // clang-format off
    right_4 << 1.75, 2.25, 2.75, 2.5, -9999., 
               2.75, 2.25, 1.75, 2.25, -9999., 
               2.75, 2.75, 2.25, 2.25, -9999., 
               2., 2.75, 3., 3., -9999., 
               -9999., -9999., -9999., -9999., -9999.;
    // clang-format on

    imgs_right.push_back(right_2);
    imgs_right.push_back(right_3);
    imgs_right.push_back(right_4);

    // Biggest shape with subpix=2
    CostVolumeSize cv_size = CostVolumeSize(5, 5, 5, 9);
    P2d::VectorD cv_values = P2d::VectorD::Zero(cv_size.size());

    // Largest disparity ranges with subpix=2
    P2d::VectorD disp_range_row(5);
    disp_range_row << -1.0, -0.5, 0.0, 0.5, 1.0;
    P2d::VectorD disp_range_col(9);
    disp_range_col << -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0;

    // Offset between cv and image first points
    int offset_cv_img_row = 0;
    int offset_cv_img_col = 0;

    // Window size
    int window_size = 3;
    // [step_row, step_col]
    Eigen::Vector2i step;
    step << 1, 1;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 2, 2);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt <<
    // d_col    -2          -1.5         -1                               d_row
          INIT_VALUE_CV, INIT_VALUE_CV, 0.22478750958935989,              // -1
        //     -0.5                      0              0.5
        0.0072146184745174313, 0.22478750958935989, 0.32440939317155548,  // -1
        //        1                      1.5             2
        0.072780225783732888,       INIT_VALUE_CV,  INIT_VALUE_CV,        //-1


    // d_col   -2           -1.5         -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.072780225783732666,               // -0.5
        //  -0.5                          0              0.5
        0.0025652873671377918, 0.091091007603791629, 0.14556045156746578, // -0.5
        //   1                            1.5            2
        0.0072146184745172093,       INIT_VALUE_CV, INIT_VALUE_CV,       // -0.5


    // d_col  -2            -1.5         -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.22478750958935989,                // 0
        //  -0.5                          0               0.5
        0.091091007603791851, 0.10218717094933361, 0.091091007603791851,  // 0
        //  1                             1.5              2
        0.37887883713522919,         INIT_VALUE_CV,   INIT_VALUE_CV,      // 0


    // d_col -2              -1.5        -1                               d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.10218717094933361,                // 0.5
        //  -0.5                          0               0.5
        0.0072146184745172093, 0.072780225783732, 0.091091007603791851,   // 0.5
        //  1                             1.5             2
        0.10218717094933316,         INIT_VALUE_CV,  INIT_VALUE_CV,       // 0.5


    // d_col -2               -1.5        -1                              d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.0072146184745172093,              // 1
        //  -0.5                          0               0.5
        0.072780225783732666, 0.22478750958935989, 0.10218717094933316,   // 1
        //  1                             1.5             2
        0.0072146184745172093,       INIT_VALUE_CV,  INIT_VALUE_CV;       // 1
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  };

  SUBCASE("Cost surface with ROI") {
    // Smallest shape with ROI
    CostVolumeSize cv_size = CostVolumeSize(3, 2, 3, 5);
    cv_values = P2d::VectorD::Zero(cv_size.size());

    // When ROI is used, we can have an offset between image and cv first index
    // to be sure to compute the first point of ROI
    offset_cv_img_row = 2;
    offset_cv_img_col = 3;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 0, 0);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << 0.22478750958935989, 0.22478750958935989, 0.072780225783732888, INIT_VALUE_CV, INIT_VALUE_CV, 
                       0.22478750958935989, 0.10218717094933361, 0.37887883713522919, INIT_VALUE_CV, INIT_VALUE_CV, 
                       0.0072146184745172093, 0.22478750958935989, 0.0072146184745172093, INIT_VALUE_CV, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface with ROI and step=[1,2]") {
    // Smallest shape with ROI
    CostVolumeSize cv_size = CostVolumeSize(5, 2, 3, 5);
    cv_values = P2d::VectorD::Zero(cv_size.size());

    // When ROI is used, we can have an offset between image and cv first index
    // to be sure to compute the first point of ROI
    offset_cv_img_col = 1;

    step << 1, 2;
    ;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, cv_size, disp_range_row,
                             disp_range_col, offset_cv_img_row, offset_cv_img_col, window_size,
                             step, no_data);

    P2d::VectorD cost_surface = get_cost_surface(cv_values, cv_size, 1, 0);

    P2d::VectorD cost_surface_gt(disp_range_row.size() * disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       INIT_VALUE_CV, INIT_VALUE_CV, 0.22478750958935989,  0.22478750958935989, 0.072780225783732888, 
                       INIT_VALUE_CV, INIT_VALUE_CV,  0.22478750958935989,  0.10218717094933361,  0.37887883713522919;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<P2d::MatrixD>(cost_surface, cost_surface_gt);
  }
}