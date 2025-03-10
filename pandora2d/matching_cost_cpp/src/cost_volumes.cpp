/* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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

#include <algorithm>

#include "cost_volumes.hpp"
#include "mutual_information.hpp"

/**
 * @brief Get the matching cost window
 *
 * @param img image
 * @param window_size size of the matching cost window
 * @param index_row row index of the center of the window
 * @param index_col col index of the center of the window
 * @return t_MatrixD
 */

t_MatrixD get_window(const t_MatrixD& img, int window_size, int index_row, int index_col) {
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

  return img.block(start_row, start_col, nb_rows_window, nb_cols_window);
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
int interpolated_right_image_index(int subpix, double disp_row, double disp_col) {
  // x - std::floor(x) is equivalent to x%1 in python
  return (subpix * subpix * (disp_row - std::floor(disp_row))) +
         subpix * (disp_col - std::floor(disp_col));
};

/**
 * @brief Check if en Eigen matrix contains an element given as parameter
 *
 * @param matrix Eigen matrix
 * @param element to check
 * @return true
 * @return false
 */
bool contains_element(const t_MatrixD& matrix, double element) {
  if (std::isnan(element)) {
    return (matrix.array().isNaN()).any();
  } else {
    return (matrix.array() == element).any();
  }
}

/**
 * @brief Compute the cost values
 *
 * @param left image
 * @param right list of right images
 * @param cv_values initialized cost values
 * @param cv_shape cost volumes 4D shape
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 * @param no_data no data value in img
 *
 * @throws std::invalid_argument if provided method is not known
 *
 * @return t_VectorD computed cost values
 */
void compute_cost_volumes_cpp(const t_MatrixD& left,
                              const std::vector<t_MatrixD>& right,
                              Eigen::Ref<t_VectorD> cv_values,
                              const Eigen::Vector4i& cv_shape,
                              const t_VectorD& disp_range_row,
                              const t_VectorD& disp_range_col,
                              int offset_cv_img_row,
                              int offset_cv_img_col,
                              int window_size,
                              const Eigen::Vector2i& step,
                              const double no_data) {
  int nb_rows_cv = cv_shape[0];
  int nb_cols_cv = cv_shape[1];
  int nb_d_rows_cv = cv_shape[2];
  int nb_d_cols_cv = cv_shape[3];

  t_MatrixD window_left;
  t_MatrixD window_right;

  int subpix = sqrt(right.size());

  // ind_cv corresponds to:
  // row * nb_d_cols_cv * nb_d_rows_cv * nb_cols_cv + col * nb_d_cols_cv * nb_d_rows_cv
  // Computation to be changed when criteria will be used in mutual information
  int ind_cv = 0;

  for (int row = 0; row < nb_rows_cv; ++row) {
    for (int col = 0; col < nb_cols_cv; ++col)

    {
      // Window computation for left image for point (row,col)
      window_left = get_window(left, window_size, offset_cv_img_row + row * step[0],
                               offset_cv_img_col + col * step[1]);

      auto left_has_no_data = contains_element(window_left, no_data);

      for (int d_row = 0; d_row < nb_d_rows_cv; ++d_row) {
        for (int d_col = 0; d_col < nb_d_cols_cv; ++d_col, ind_cv++) {
          int index_right =
              interpolated_right_image_index(subpix, disp_range_row[d_row], disp_range_col[d_col]);

          // Window computation for right image for point (row+d_row,col+d_col)
          window_right =
              get_window(right[index_right], window_size,
                         offset_cv_img_row + row * step[0] + floor(disp_range_row[d_row]),
                         offset_cv_img_col + col * step[1] + floor(disp_range_col[d_col]));

          // To compute the similarity value, the left and right windows must have the same
          // size.
          if ((window_right.size() != window_left.size())) {
            continue;
          }
          // To be replaced with a condition on criteria map
          if (contains_element(window_right, no_data) or left_has_no_data) {
            continue;
          }
          cv_values[ind_cv] = calculate_mutual_information(window_left, window_right);
        }
      }
    }
  }
};
