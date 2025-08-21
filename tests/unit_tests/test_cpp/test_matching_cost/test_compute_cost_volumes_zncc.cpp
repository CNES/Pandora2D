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
This module contains tests associated to zncc computation.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <pybind11/embed.h>
#include "compute_cost_volumes.hpp"
#include "conftest.hpp"
#include "cost_volume.hpp"
#include "global_conftest.hpp"

constexpr double INIT_VALUE_CV = 0.0;  ///< initial value used to fill the cv

template <typename cost_volume_type, typename cost_surface_type>
struct TypePairCV {
  using CostVolumeType = cost_volume_type;
  using CostSurfaceType = cost_surface_type;
};

TYPE_TO_STRING_AS("Float", TypePairCV<float, P2d::Matrixf>);
TYPE_TO_STRING_AS("Double", TypePairCV<double, P2d::MatrixD>);

TEST_CASE_TEMPLATE("Test compute_cost_volumes_cpp method with zncc",
                   T,
                   TypePairCV<float, P2d::Matrixf>,
                   TypePairCV<double, P2d::MatrixD>) {
  using CostVolumeType = typename T::CostVolumeType;
  using CostSurfaceType = typename T::CostSurfaceType;
  // Left image
  P2d::Matrixf img_left(5, 5);
  // clang-format off
  img_left << 1.0, 2.0, 3.0, 4.0, 5.0, 
              6.0, 7.0, 8.0, 9.0, 10.0, 
              11.0, 12.0, 13.0, 14.0, 15.0, 
              16.0, 17.0, 18.0, 19.0, 20.0, 
              21.0, 22.0, 23.0, 24.0, 25.0;
  // clang-format on

  // Right image
  std::vector<P2d::Matrixf> imgs_right;

  P2d::Matrixf right_1(5, 5);
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
  py::scoped_interpreter guard{};
  std::vector<CostVolumeType> zeros(cv_size.size(), 0.);
  py::array_t<CostVolumeType> cv_values(
      {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, zeros.data());

  // Initialized pixel
  Position2D pixel = Position2D();

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

  // correlation method
  std::string matching_cost_method = "zncc";

  SUBCASE("Left border") {
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/top_left_criteria.bin", cv_size);

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // We are on the top left edge of the image, so the criterion P2D_LEFT_BORDER
    // is raised for the entire cost surface
    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, 
                       INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface of center point") {
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/center_criteria.bin", cv_size);

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    pixel = Position2D(2, 2);
    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, 0.49029034, -0.49029034, -0.25129602, INIT_VALUE_CV,
                       INIT_VALUE_CV, 0.16048518,  0.5204165 ,  0.14563793, INIT_VALUE_CV, 
                       INIT_VALUE_CV, -0.39090492,  0.45760432,  0.39291264, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface with step_row=2, step_col=3") {
    // Smaller shape with step=[2,3]
    CostVolumeSize cv_size = CostVolumeSize(3, 2, 3, 5);
    std::vector<CostVolumeType> zeros(cv_size.size(), 0.);
    py::array_t<CostVolumeType> cv_values(
        {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, zeros.data());

    // Criteria values
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/step_[2,3]_criteria.bin", cv_size);

    step << 2, 3;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    pixel = Position2D(1, 1);
    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // clang-format off
    cost_surface_gt << 0.49029034, -0.49029034, -0.25129602, INIT_VALUE_CV, INIT_VALUE_CV, 
                       0.16048518,  0.5204165 ,  0.14563793, INIT_VALUE_CV, INIT_VALUE_CV, 
                       -0.39090492,  0.45760432,  0.39291264, INIT_VALUE_CV, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  }
  SUBCASE("Cost surface with subpix=2") {
    // When subpix=2, we have 4 right images
    P2d::Matrixf right_2(5, 5);

    // clang-format off
    right_2 << 1.5, 2.5, 3.5, 3., -9999., 
               2., 2., 2., 2., -9999., 
               3.5, 2.5, 1.5, 2.5, -9999., 
               2., 3., 3., 2., -9999., 
               2., 2.5, 3., 4., -9999.;
    // clang-format on

    P2d::Matrixf right_3(5, 5);
    // clang-format off
    right_3 << 1.5, 2., 2.5, 3., 2., 
               3., 2.5, 2., 1.5, 3., 
               2.5, 3., 2.5, 2., 2.5, 
               1., 3., 2.5, 3.5, 2.5, 
               -9999., -9999., -9999., -9999., -9999.;
    // clang-format on

    P2d::Matrixf right_4(5, 5);
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
    std::vector<CostVolumeType> zeros(cv_size.size(), 0.);
    py::array_t<CostVolumeType> cv_values(
        {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, zeros.data());

    // Criteria values
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/subpix_2_criteria.bin", cv_size);

    // Largest disparity ranges with subpix=2
    P2d::VectorD disp_range_row(5);
    disp_range_row << -1.0, -0.5, 0.0, 0.5, 1.0;
    P2d::VectorD disp_range_col(9);
    disp_range_col << -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    pixel = Position2D(2, 2);
    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // When a subpix other than 1 is used, the method used to calculate the criteria is nearest
    // neighbor. In this case, for disp_col=1.5, we round up to disp_col=2.
    // clang-format off
    cost_surface_gt <<
    // d_col    -2          -1.5         -1             d_row
          INIT_VALUE_CV, INIT_VALUE_CV, 0.490290338,    // -1
        //     -0.5          0            0.5
        -1.20679683e-15, -0.490290338, -0.565266864,    // -1
        //        1          1.5          2
        -0.251296017,   INIT_VALUE_CV, INIT_VALUE_CV,   //-1


    // d_col   -2           -1.5         -1             d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.582551728,      // -0.5
        //  -0.5             0            0.5
        0.305714799, -0.0566138517, -0.138675049,       // -0.5
        //   1               1.5          2
        -0.113227703,  INIT_VALUE_CV, INIT_VALUE_CV,    // -0.5


    // d_col  -2            -1.5         -1             d_row
        INIT_VALUE_CV, INIT_VALUE_CV, 0.160485185,      // 0
        //  -0.5             0            0.5
        0.391427690,  0.520416500,  0.502592034,        // 0
        //  1                1.5          2
        0.145637932,   INIT_VALUE_CV, INIT_VALUE_CV,    // 0


    // d_col -2              -1.5        -1             d_row
        INIT_VALUE_CV, INIT_VALUE_CV, -0.200711599,     // 0.5
        //  -0.5              0           0.5
        0.301229742,  0.624037721,  0.820014299,        // 0.5
        //  1                 1.5         2
        0.533787745, INIT_VALUE_CV, INIT_VALUE_CV,      // 0.5


    // d_col -2               -1.5        -1            d_row
        INIT_VALUE_CV, INIT_VALUE_CV, -0.390904915,     // 1
        //  -0.5               0           0.5
        INIT_VALUE_CV, 0.457604315,  0.620496596,       // 1
        //  1                  1.5         2
        0.392912639, INIT_VALUE_CV, INIT_VALUE_CV;      // 1
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  };

  SUBCASE("All points are invalid") {
    // All points of cost volumes are invalid
    std::vector<CostVolumeType> ones(cv_size.size(), 1.);
    py::array_t<CostVolumeType> criteria_values(
        {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, ones.data());

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);
    // Get a view on cv values
    auto cv_values_view = cv_values.template unchecked<4>();

    // Check that all points of cost volumes are equal to INIT_VALUE_CV
    for (ssize_t row = 0; row < cv_values_view.shape(0); ++row)
      for (ssize_t col = 0; col < cv_values_view.shape(1); ++col)
        for (ssize_t d_row = 0; d_row < cv_values_view.shape(2); ++d_row)
          for (ssize_t d_col = 0; d_col < cv_values_view.shape(3); ++d_col)
            CHECK(cv_values_view(row, col, d_row, d_col) == INIT_VALUE_CV);

    CHECK(cv_values.size() == cv_size.size());
  }

  SUBCASE("Cost surface of center point with not centered disparities") {
    // CV size
    CostVolumeSize cv_size = CostVolumeSize(5, 5, 3, 3);

    // Initialized cv values
    std::vector<CostVolumeType> zeros(cv_size.size(), 0.);
    py::array_t<CostVolumeType> cv_values(
        {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, zeros.data());

    // Criteria values
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/not_centered_disp_criteria.bin", cv_size);

    // Disparity ranges
    P2d::VectorD disp_range_row(3);
    disp_range_row << -2.0, -1.0, 0.0;
    P2d::VectorD disp_range_col(3);
    disp_range_col << 0.0, 1.0, 2.0;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    pixel = Position2D(2, 2);
    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       -0.49029034, -0.25129602, INIT_VALUE_CV,
                       0.5204165 ,  0.14563793, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  }

  SUBCASE("Cost surface with ROI and step=[1,2]") {
    // Is equivalent to ROI: {"col": {"first": 1, "last": 2},
    //                        "row": {"first": 1, "last": 2},
    //                        "margins": [1, 1, 1, 1]}

    // Left image
    P2d::Matrixf img_left(5, 4);
    // clang-format off
    img_left << 1.0, 2.0, 3.0, 4.0, 
                6.0, 7.0, 8.0, 9.0, 
                11.0, 12.0, 13.0, 14.0, 
                16.0, 17.0, 18.0, 19.0;
    // clang-format on

    // Right image
    std::vector<P2d::Matrixf> imgs_right;

    P2d::Matrixf right_1(5, 4);
    // clang-format off
    right_1 << 1., 2., 3., 4., 
              2., 2., 2., 2., 
              4., 3., 2., 1., 
              1., 3., 3., 3.;
    // clang-format on

    imgs_right.push_back(right_1);

    // Smallest shape with ROI
    CostVolumeSize cv_size = CostVolumeSize(4, 2, 3, 5);
    std::vector<CostVolumeType> zeros(cv_size.size(), 0.);
    py::array_t<CostVolumeType> cv_values(
        {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row, cv_size.nb_disp_col}, zeros.data());

    // Criteria values
    py::array_t<uint8_t> criteria_values =
        load_criteria_dataarray(data_path + "/data/roi_step_[1,2]_criteria.bin", cv_size);

    // When ROI is used with a step different from 1,
    // we can have an offset between image and cv first index
    // to be sure to compute the first point of ROI
    offset_cv_img_col = 1;

    step << 1, 2;

    compute_cost_volumes_cpp(img_left, imgs_right, cv_values, criteria_values, cv_size,
                             disp_range_row, disp_range_col, offset_cv_img_row, offset_cv_img_col,
                             window_size, step, matching_cost_method);

    pixel = Position2D(1, 0);
    CostSurfaceType cost_surface = get_cost_surface<CostVolumeType, CostVolumeType>(
        cv_values, position2d_to_index(pixel, cv_size), cv_size);

    CostSurfaceType cost_surface_gt(disp_range_row.size(), disp_range_col.size());

    // clang-format off
    cost_surface_gt << INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV, INIT_VALUE_CV,
                       INIT_VALUE_CV, INIT_VALUE_CV, 0.49029055, -0.49028968, INIT_VALUE_CV,
                       INIT_VALUE_CV, INIT_VALUE_CV, 0.16048510, 0.52041649, INIT_VALUE_CV;
    // clang-format on

    CHECK(cv_values.size() == cv_size.size());
    check_inside_eigen_element<CostSurfaceType>(cost_surface, cost_surface_gt);
  }
};