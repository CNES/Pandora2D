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
#include <map>

#include <iostream>
#include "cost_volume.hpp"
#include "mutual_information.hpp"
#include "zncc.hpp"

/**
 * @brief Get the matching cost window
 *
 * @param img image
 * @param window_size size of the matching cost window
 * @param index_row row index of the center of the window
 * @param index_col col index of the center of the window
 * @return P2d::MatrixD
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> get_window(const P2d::Matrixf& img,
                                                            int window_size,
                                                            int index_row,
                                                            int index_col) {
  int offset = static_cast<int>(window_size / 2);

  // Get first row and column of the window
  int start_row = std::max(0, index_row - offset);
  int start_col = std::max(0, index_col - offset);

  // Get last row and column of the window
  int nb_rows_img = static_cast<int>(img.rows());
  int nb_cols_img = static_cast<int>(img.cols());
  int end_row = std::min(nb_rows_img - 1, index_row + offset);
  int end_col = std::min(nb_cols_img - 1, index_col + offset);

  // if the window is out of the image,
  // nb_rows_window or nb_cols_window are < 0
  // in this case we return an empty window
  int nb_rows_window = std::max(0, end_row - start_row + 1);
  int nb_cols_window = std::max(0, end_col - start_col + 1);

  return img.block(start_row, start_col, nb_rows_window, nb_cols_window).template cast<T>();
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
 * @brief
 *
 * @param method correlation method
 * @param left_image left image
 * @param right_image right image
 * @return correlation value
 */
template <typename T>
T calculate_correlation(const std::string& method,
                        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& left_image,
                        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& right_image) {
  std::map<std::string, std::function<T(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&,
                                        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&)>>
      method_map = {{"mutual_information", calculate_mutual_information<T>},
                    {"zncc", calculate_zncc<T>}};
  auto it = method_map.find(method);
  if (it != method_map.end()) {
    return it->second(left_image, right_image);
  } else {
    throw std::invalid_argument("Unknown correlation method: " + method);
  }
}

/**
 * @brief Compute the cost values
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
 * @param no_data no data value in img
 * @param matching_cost_method correlation method
 *
 * @throws std::invalid_argument if provided method is not known
 *
 */
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
                              const double no_data,
                              const std::string matching_cost_method) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> left_window;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> right_window;

  if (matching_cost_method == "zncc") {
    std::cerr << "ZNCC method not yet implemented in C++." << std::endl;
    return;
  }

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
              calculate_correlation<T>(matching_cost_method, left_window, right_window);
        }
      }
    }
  }
};

#endif