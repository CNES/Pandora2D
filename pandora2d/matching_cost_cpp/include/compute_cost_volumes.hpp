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
This module contains functions associated to the computation of cost volumes in cpp.
*/

#ifndef COMPUTE_COST_VOLUMES_HPP
#define COMPUTE_COST_VOLUMES_HPP

#include <Eigen/Dense>
#include <functional>
#include <map>

#include "cost_volume.hpp"
#include "mutual_information.hpp"
#include "zncc.hpp"

/**
 * @brief Get the matching cost window
 *
 * @param image image
 * @param window_size size of the matching cost window
 * @param index_row row index of the center of the window
 * @param index_col col index of the center of the window
 * @return P2d::MatrixD or P2d::Matrixf
 */
template <typename T>
P2d::MatrixX<T> get_window(const P2d::Matrixf& image,
                           int window_size,
                           int index_row,
                           int index_col) {
  int offset = static_cast<int>(window_size / 2);

  // Get first row and column of the window
  int start_row = std::max(0, index_row - offset);
  int start_col = std::max(0, index_col - offset);

  // Get last row and column of the window
  int nb_rows_img = static_cast<int>(image.rows());
  int nb_cols_img = static_cast<int>(image.cols());
  int end_row = std::min(nb_rows_img - 1, index_row + offset);
  int end_col = std::min(nb_cols_img - 1, index_col + offset);

  // if the window is out of the image,
  // nb_rows_window or nb_cols_window are < 0
  // in this case we return an empty window
  int nb_rows_window = std::max(0, end_row - start_row + 1);
  int nb_cols_window = std::max(0, end_col - start_col + 1);

  return image.block(start_row, start_col, nb_rows_window, nb_cols_window).template cast<T>();
}

/**
 * @brief Get the index corresponding to the correct interpolated right image
 * according to subpix value
 *
 * @param subpix value
 * @param disp_row value
 * @param disp_col value
 * @return int right index
 */
int interpolated_right_image_index(int subpix, double disp_row, double disp_col);

/**
 * @brief Returns true if there are only elements other than 0 in the vector
 *
 * @param mat
 * @return true or false
 */
bool all_non_zero_elements(const P2d::MatrixUI& mat);

/**
 * @brief Compute the cost values with mutual information
 *
 * @param left image
 * @param right list of right images
 * @param cv_values initialized cost values
 * @param criteria_values criteria values
 * @param cv_size : cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 */
template <typename T>
void compute_mutual_information_cv(const P2d::Matrixf& left,
                                   const std::vector<P2d::Matrixf>& right,
                                   py::array_t<T>& cv_values,
                                   const py::array_t<uint8_t>& criteria_values,
                                   CostVolumeSize& cv_size,
                                   const P2d::VectorD& disp_range_row,
                                   const P2d::VectorD& disp_range_col,
                                   int offset_cv_img_row,
                                   int offset_cv_img_col,
                                   int window_size,
                                   const Eigen::Vector2i& step) {
  P2d::MatrixX<T> left_window;
  P2d::MatrixX<T> right_window;

  int subpix = sqrt(right.size());

  // ind_cv corresponds to:
  // row * cv_size.nb_disps() * nb_col + col * cv_size.nb_disps()
  int ind_cv = 0;

  int cost_surface_size = cv_size.nb_disps();

  P2d::MatrixUI criteria_cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);

  for (std::size_t row = 0; row < cv_size.nb_row; ++row) {
    for (std::size_t col = 0; col < cv_size.nb_col; ++col)

    {
      // Get criteria cost surface to check if the entire cost surface is invalid
      criteria_cost_surface = get_cost_surface<uint8_t, uint8_t>(criteria_values, ind_cv, cv_size);

      // If the entire cost surface is invalid, we do not compute cost volumes for this point
      if (all_non_zero_elements(criteria_cost_surface)) {
        ind_cv += cost_surface_size;
        continue;
      }

      // Window computation for left image for point (row,col)
      left_window = get_window<T>(left, window_size, offset_cv_img_row + row * step[0],
                                  offset_cv_img_col + col * step[1]);

      for (std::size_t d_row = 0; d_row < cv_size.nb_disp_row; ++d_row) {
        for (std::size_t d_col = 0; d_col < cv_size.nb_disp_col; ++d_col, ind_cv++) {
          // Get a view on criteria value at point (row, col, d_row, d_col)
          auto criteria_value_view = criteria_values.unchecked<4>();
          uint8_t criteria_value = criteria_value_view(row, col, d_row, d_col);

          // Get access to cv_value at point (row, col, d_row, d_col) to be able to modify it
          auto cv_mutable_view = cv_values.template mutable_unchecked<4>();

          if (criteria_value != 0) {
            continue;
          }
          int index_right =
              interpolated_right_image_index(subpix, disp_range_row[d_row], disp_range_col[d_col]);

          // Window computation for right image for point (row+d_row,col+d_col)
          right_window =
              get_window<T>(right[index_right], window_size,
                            offset_cv_img_row + row * step[0] + floor(disp_range_row[d_row]),
                            offset_cv_img_col + col * step[1] + floor(disp_range_col[d_col]));

          cv_mutable_view(row, col, d_row, d_col) =
              calculate_mutual_information<T>(left_window, right_window);
        }
      }
    }
  }
};

/**
 * @brief Compute the cost values with zncc
 *
 * @param left image
 * @param right list of right images
 * @param cv_values initialized cost values
 * @param criteria_values criteria values
 * @param cv_size : cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 */
template <typename T>
void compute_zncc_cv(const P2d::Matrixf& left,
                     const std::vector<P2d::Matrixf>& right,
                     py::array_t<T>& cv_values,
                     const py::array_t<uint8_t>& criteria_values,
                     CostVolumeSize& cv_size,
                     const P2d::VectorD& disp_range_row,
                     const P2d::VectorD& disp_range_col,
                     int offset_cv_img_row,
                     int offset_cv_img_col,
                     int window_size,
                     const Eigen::Vector2i& step) {
  const int half_window = floor(window_size / 2);
  int subpix = sqrt(right.size());

  // Compute left integral images
  // Computation is done in double type to avoid rounding errors
  P2d::MatrixX<double> integral_left, integral_left_sq;
  compute_integral_image<double>(left, integral_left, integral_left_sq);

  // Initialize right integral images
  P2d::MatrixX<double> integral_right, integral_right_sq, integral_cross;

  // Declaration of variables used in for loop
  auto cv_mutable_view = cv_values.template mutable_unchecked<4>();
  auto criteria_value_view = criteria_values.template unchecked<4>();

  int disp_row_value;
  int disp_col_value;
  P2d::Matrixf shifted_right;

  int left_win_row_center;
  int left_win_col_center;
  int top_row;
  int left_col;
  int bottom_row;
  int right_col;

  double zncc;

  for (std::size_t d_row = 0; d_row < cv_size.nb_disp_row; ++d_row) {
    for (std::size_t d_col = 0; d_col < cv_size.nb_disp_col; ++d_col) {
      int index_right =
          interpolated_right_image_index(subpix, disp_range_row[d_row], disp_range_col[d_col]);

      disp_row_value = floor(disp_range_row[d_row]);
      disp_col_value = floor(disp_range_col[d_col]);

      // Compute shifted right image according to disparities
      shifted_right = shift_image(right[index_right], disp_row_value, disp_col_value);

      // Computed right and cross integral images
      // Computation is done in double type to avoid rounding errors
      compute_right_integrals<double>(left, shifted_right, integral_right, integral_right_sq,
                                      integral_cross);

      for (std::size_t row = 0; row < cv_size.nb_row; ++row) {
        for (std::size_t col = 0; col < cv_size.nb_col; ++col) {
          if (criteria_value_view(row, col, d_row, d_col) != 0) {
            continue;
          }

          left_win_row_center = offset_cv_img_row + row * step[0];
          left_win_col_center = offset_cv_img_col + col * step[1];

          top_row = left_win_row_center - half_window;
          left_col = left_win_col_center - half_window;
          bottom_row = top_row + window_size - 1;
          right_col = left_col + window_size - 1;

          // Computation is done in double type to avoid rounding errors
          zncc =
              calculate_zncc(integral_left, integral_left_sq, integral_right, integral_right_sq,
                             integral_cross, top_row, left_col, bottom_row, right_col, window_size);

          cv_mutable_view(row, col, d_row, d_col) = static_cast<T>(zncc);
        }
      }
    }
  }
}

template <typename T>
using ComputeFunction = std::function<void(const P2d::Matrixf&,
                                           const std::vector<P2d::Matrixf>&,
                                           py::array_t<T>&,
                                           const py::array_t<uint8_t>&,
                                           CostVolumeSize&,
                                           const P2d::VectorD&,
                                           const P2d::VectorD&,
                                           int,
                                           int,
                                           int,
                                           const Eigen::Vector2i&)>;

template <typename T>
void compute_cost_volumes_cpp(const P2d::Matrixf& left,
                              const std::vector<P2d::Matrixf>& right,
                              py::array_t<T>& cv_values,
                              const py::array_t<uint8_t>& criteria_values,
                              CostVolumeSize& cv_size,
                              const P2d::VectorD& disp_range_row,
                              const P2d::VectorD& disp_range_col,
                              int offset_cv_img_row,
                              int offset_cv_img_col,
                              int window_size,
                              const Eigen::Vector2i& step,
                              const std::string& method) {
  static const std::map<std::string, ComputeFunction<T>> method_map = {
      {"mutual_information", compute_mutual_information_cv<T>}, {"zncc", compute_zncc_cv<T>}};

  auto it = method_map.find(method);
  if (it != method_map.end()) {
    it->second(left, right, cv_values, criteria_values, cv_size, disp_range_row, disp_range_col,
               offset_cv_img_row, offset_cv_img_col, window_size, step);
  } else {
    throw std::invalid_argument("Unknown correlation method: " + method);
  }
}
#endif