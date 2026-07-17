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
This module contains functions associated to the computation of cost volumes in cpp.
*/

#ifndef COMPUTE_COST_VOLUMES_HPP
#define COMPUTE_COST_VOLUMES_HPP

#include <Eigen/Dense>
#include <functional>
#include <map>
#include <variant>

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
 * @return P2d::Matrixf
 */
P2d::Matrixf get_window(const Eigen::Ref<const P2d::Matrixf>& image,
                        int window_size,
                        int index_row,
                        int index_col);

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
 * @brief Get the index corresponding to the disp_val value in the disparity range
 *
 * @param disp_range disparity range
 * @param disp_val disparity value for which we want to find the index
 * @return int index of the disparity value in the disparity range
 */
int disparity_index(const P2d::VectorD& disp_range, float disp_val);

/**
 * @brief Correlator type: can be either a ZnccCorrelator or a MutualInformationCorrelator
 */
template <typename T>
using Correlator = std::variant<ZnccOpt2Correlator<T>, MutualInformationCorrelator<T>>;

/**
 * @brief Return the correlator corresponding to the given method
 *
 * @param method correlation method ("mutual_information" or "zncc-optim-2")
 * @return Correlator<T> pointer to the matching cost function
 */
template <typename T>
Correlator<T> make_correlator(const std::string& method) {
  if (method == "zncc-optim-2") {
    return ZnccOpt2Correlator<T>{};
  }
  if (method == "mutual_information") {
    return MutualInformationCorrelator<T>{};
  }
  throw std::invalid_argument("Unknown correlation method: " + method);
}

/**
 * @brief Compute the cost values loop with compatible metrics
 *
 * @param left image
 * @param min_disp_row minimum row disparity grid
 * @param max_disp_row maximum row disparity grid
 * @param min_disp_col minimum col disparity grid
 * @param max_disp_col maximum col disparity grid
 * @param right list of right images
 * @param cv_values 1D initialized cost values
 * @param criteria_values 1D criteria values
 * @param cv_size cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 * @param matching_cost_method is the method used within the loop
 */
template <typename T>
void compute_cost_volumes_loop(const Eigen::Ref<const P2d::Matrixf>& left,
                               const py::array_t<float>& min_disp_row,
                               const py::array_t<float>& max_disp_row,
                               const py::array_t<float>& min_disp_col,
                               const py::array_t<float>& max_disp_col,
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
                               const std::string matching_cost_method) {
  auto correlator = make_correlator<T>(matching_cost_method);

  P2d::Matrixf left_window;
  P2d::Matrixf right_window;
  int ind_cv;

  auto criteria_value_view = criteria_values.unchecked<4>();
  auto cv_mutable_view = cv_values.template mutable_unchecked<4>();
  auto min_disp_row_view = min_disp_row.unchecked<2>();
  auto max_disp_row_view = max_disp_row.unchecked<2>();
  auto min_disp_col_view = min_disp_col.unchecked<2>();
  auto max_disp_col_view = max_disp_col.unchecked<2>();

  int subpix = sqrt(right.size());
  int cost_surface_size = cv_size.nb_disps();
  P2d::MatrixUI criteria_cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);

  for (std::size_t row = 0; row < cv_size.nb_row; ++row) {
    for (std::size_t col = 0; col < cv_size.nb_col; ++col) {
      // Compute ind_cv
      ind_cv = (row * cv_size.nb_col + col) * cost_surface_size;

      // Get criteria cost surface to check if the entire cost surface is invalid
      criteria_cost_surface = get_cost_surface<uint8_t, uint8_t>(criteria_values, ind_cv, cv_size);

      // If the entire cost surface is invalid, we do not compute cost volumes for this point
      if (all_non_zero_elements(criteria_cost_surface)) {
        continue;
      }

      // Get local disparity range for pixel (row, col)
      int d_row_start = disparity_index(disp_range_row, min_disp_row_view(row, col));
      int d_row_end = disparity_index(disp_range_row, max_disp_row_view(row, col));
      int d_col_start = disparity_index(disp_range_col, min_disp_col_view(row, col));
      int d_col_end = disparity_index(disp_range_col, max_disp_col_view(row, col));

      // Window computation for left image for point (row,col)
      left_window = get_window(left, window_size, offset_cv_img_row + row * step[0],
                               offset_cv_img_col + col * step[1]);

      // Prepare the currently selected correlator using the left image window.
      // correlator is either a ZnccOpt2Correlator<T> or a MutualInformationCorrelator<T>,
      // decided by the matching_cost_method parameter. std::visit inspects which one is actually
      // stored and calls prepare_left_window() on it with the matching concrete type --> equivalent
      // to a manual if/else dispatch on the stored type, but generated by the compiler.
      // clang-format off
      std::visit(
          [&](auto& correlator_impl) {
            correlator_impl.prepare_left_window(left_window);
          },
          correlator);
      // clang-format on

      for (int d_row = d_row_start; d_row <= d_row_end; ++d_row) {
        for (int d_col = d_col_start; d_col <= d_col_end; ++d_col) {
          // Check criteria value for point (row, col) and disparity (d_row, d_col)
          uint8_t criteria_value = criteria_value_view(row, col, d_row, d_col);
          if (criteria_value != 0) {
            continue;
          }

          int index_right =
              interpolated_right_image_index(subpix, disp_range_row[d_row], disp_range_col[d_col]);

          // Window computation for right image for point (row+d_row,col+d_col)
          right_window =
              get_window(right[index_right], window_size,
                         offset_cv_img_row + row * step[0] + floor(disp_range_row[d_row]),
                         offset_cv_img_col + col * step[1] + floor(disp_range_col[d_col]));

          // Compute the cost value for point (row, col) and disparity (d_row, d_col)
          cv_mutable_view(row, col, d_row, d_col) = std::visit(
              [&](auto& correlator_impl) { return correlator_impl(right_window); }, correlator);
        }
      }
    }
  }
};

/**
 * @brief Compute the cost values with zncc with optimisation 1 using integral images
 *
 * @param left image
 * @param min_disp_row minimum row disparity grid
 * @param max_disp_row maximum row disparity grid
 * @param min_disp_col minimum col disparity grid
 * @param max_disp_col maximum col disparity grid
 * @param right list of right images
 * @param cv_values 1D initialized cost values
 * @param criteria_values 1D criteria values
 * @param cv_size cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 * @param matching_cost_method unused, template of the main function
 */
template <typename T>
void compute_zncc_cv_opt1(const Eigen::Ref<const P2d::Matrixf>& left,
                          const py::array_t<float>& min_disp_row,
                          const py::array_t<float>& max_disp_row,
                          const py::array_t<float>& min_disp_col,
                          const py::array_t<float>& max_disp_col,
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
                          const std::string matching_cost_method) {
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
  auto min_disp_row_view = min_disp_row.unchecked<2>();
  auto max_disp_row_view = max_disp_row.unchecked<2>();
  auto min_disp_col_view = min_disp_col.unchecked<2>();
  auto max_disp_col_view = max_disp_col.unchecked<2>();

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
          if (criteria_value_view(row, col, d_row, d_col) != 0)
            continue;

          // Skip if this disparity is outside the local range for (row, col)
          if (disp_range_row[d_row] < min_disp_row_view(row, col) ||
              disp_range_row[d_row] > max_disp_row_view(row, col) ||
              disp_range_col[d_col] < min_disp_col_view(row, col) ||
              disp_range_col[d_col] > max_disp_col_view(row, col))
            continue;

          left_win_row_center = offset_cv_img_row + row * step[0];
          left_win_col_center = offset_cv_img_col + col * step[1];
          top_row = left_win_row_center - half_window;
          left_col = left_win_col_center - half_window;
          bottom_row = top_row + window_size - 1;
          right_col = left_col + window_size - 1;

          // Computation is done in double type to avoid rounding errors
          // Optimisation v1 is called, using integral images
          zncc = calculate_zncc_opt1(integral_left, integral_left_sq, integral_right,
                                     integral_right_sq, integral_cross, top_row, left_col,
                                     bottom_row, right_col, window_size);

          cv_mutable_view(row, col, d_row, d_col) = static_cast<T>(zncc);
        }
      }
    }
  }
}

/**
 * @brief Methods used to compute the cost values
 */
template <typename T>
using ComputeFunction = std::function<void(const Eigen::Ref<const P2d::Matrixf>&,
                                           const py::array_t<float>&,
                                           const py::array_t<float>&,
                                           const py::array_t<float>&,
                                           const py::array_t<float>&,
                                           const std::vector<P2d::Matrixf>&,
                                           py::array_t<T>&,
                                           const py::array_t<uint8_t>&,
                                           CostVolumeSize&,
                                           const P2d::VectorD&,
                                           const P2d::VectorD&,
                                           int,
                                           int,
                                           int,
                                           const Eigen::Vector2i&,
                                           std::string)>;

/**
 * @brief Compute the cost values with method given as parameter
 *
 * @param left image
 * @param min_disp_row minimum row disparity grid
 * @param max_disp_row maximum row disparity grid
 * @param min_disp_col minimum col disparity grid
 * @param max_disp_col maximum col disparity grid
 * @param right list of right images
 * @param cv_values 1D initialized cost values
 * @param criteria_values 1D criteria values
 * @param cv_size cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 * @param method method used to compute cost values
 */
template <typename T>
void compute_cost_volumes_cpp(const Eigen::Ref<const P2d::Matrixf>& left,
                              const py::array_t<float>& min_disp_row,
                              const py::array_t<float>& max_disp_row,
                              const py::array_t<float>& min_disp_col,
                              const py::array_t<float>& max_disp_col,
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
      {"mutual_information", compute_cost_volumes_loop<T>},
      {"zncc-optim-1", compute_zncc_cv_opt1<T>},
      {"zncc-optim-2", compute_cost_volumes_loop<T>}};

  auto it = method_map.find(method);
  if (it != method_map.end()) {
    it->second(left, min_disp_row, max_disp_row, min_disp_col, max_disp_col, right, cv_values,
               criteria_values, cv_size, disp_range_row, disp_range_col, offset_cv_img_row,
               offset_cv_img_col, window_size, step, method);
  } else {
    throw std::invalid_argument("Unknown correlation method: " + method);
  }
}
#endif
