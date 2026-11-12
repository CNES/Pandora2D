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

#include <algorithm>

#include "compute_cost_volumes.hpp"
#include "mutual_information.hpp"

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
                        int index_col) {
  // Constant to use, ZERO is required for std::min / std::max
  const Eigen::Index offset = static_cast<Eigen::Index>(window_size) / 2;
  const Eigen::Index zero = 0;

  // Get first row and column of the window
  // First cast to Index
  Eigen::Index eigen_index_row = static_cast<Eigen::Index>(index_row);
  Eigen::Index eigen_index_col = static_cast<Eigen::Index>(index_col);
  Eigen::Index start_row = std::max(zero, eigen_index_row - offset);
  Eigen::Index start_col = std::max(zero, eigen_index_col - offset);

  // Get last row and column of the window
  Eigen::Index end_row = std::min(image.rows() - 1, eigen_index_row + offset);
  Eigen::Index end_col = std::min(image.cols() - 1, eigen_index_col + offset);

  // if the window is out of the image,
  // nb_rows_window or nb_cols_window are < 0
  // in this case we return an empty window
  Eigen::Index nb_rows_window = std::max(zero, end_row - start_row + 1);
  Eigen::Index nb_cols_window = std::max(zero, end_col - start_col + 1);

  return image.block(start_row, start_col, nb_rows_window, nb_cols_window);
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
 * @brief Returns true if there are only elements other than 0 in the vector
 *
 * @param mat
 * @return true or false
 */
bool all_non_zero_elements(const P2d::MatrixUI& mat) {
  return (mat.array() != 0).all();
}

/**
 * @brief Returns true if at least one element is other than 0 in the matrix
 *
 * @param mat
 * @return true or false
 */
bool any_non_zero_element(const P2d::MatrixUI& mat) {
  return (mat.array() != 0).any();
}

/**
 * @brief Compute the first and last cost-volume disparity indices for one pixel's user range.
 *
 * For variable initial disparity grids, the interval is derived from the point min/max values.
 *
 * min/max grids passed here are already expanded by refinement margins (Python side).
 * Margins are subtracted back in disparity value space to recover the user range,
 * then mapped to cost-volume indices via disparity_index().
 *
 * @param min_row minimum row disparity value in expanded range
 * @param max_row maximum row disparity value in expanded range
 * @param min_col minimum col disparity value in expanded range
 * @param max_col maximum col disparity value in expanded range
 * @param disparity_margins refinement margins on cost surface edges
 * @param subpix subpixel factor
 * @param disp_range_row cost volume row disparity range
 * @param disp_range_col cost volume col disparity range
 * @param user_d_row_start first disparity row index in user range
 * @param user_d_row_end last disparity row index in user range
 * @param user_d_col_start first disparity col index in user range
 * @param user_d_col_end last disparity col index in user range
 */
void compute_user_disparity_range_indices(float min_row,
                            float max_row,
                            float min_col,
                            float max_col,
                            const Margins& disparity_margins,
                            int subpix,
                            const P2d::VectorD& disp_range_row,
                            const P2d::VectorD& disp_range_col,
                            int& user_d_row_start,
                            int& user_d_row_end,
                            int& user_d_col_start,
                            int& user_d_col_end) {
  const float subpix_f = static_cast<float>(subpix);
  const float user_min_row = min_row + static_cast<float>(disparity_margins.up) / subpix_f;
  const float user_max_row = max_row - static_cast<float>(disparity_margins.down) / subpix_f;
  const float user_min_col = min_col + static_cast<float>(disparity_margins.left) / subpix_f;
  const float user_max_col = max_col - static_cast<float>(disparity_margins.right) / subpix_f;

  user_d_row_start = disparity_index(disp_range_row, user_min_row);
  user_d_row_end = disparity_index(disp_range_row, user_max_row);
  user_d_col_start = disparity_index(disp_range_col, user_min_col);
  user_d_col_end = disparity_index(disp_range_col, user_max_col);
}

/**
 * @brief Returns true if the cost volumes must not be computed for a point
 *
 * @param criteria_cost_surface criteria cost surface of the point
 * @param incomplete_disparity_range if true, the point is skipped as soon as one disparity of the
 * user range is invalid, otherwise only if all disparities are invalid
 * @param user_d_row_start first disparity row index in user range
 * @param user_d_row_end last disparity row index in user range
 * @param user_d_col_start first disparity col index in user range
 * @param user_d_col_end last disparity col index in user range
 * @return true or false
 */
bool is_point_skipped(const P2d::MatrixUI& criteria_cost_surface,
                      bool incomplete_disparity_range,
                      int user_d_row_start,
                      int user_d_row_end,
                      int user_d_col_start,
                      int user_d_col_end) {
  // Without the rule, a point is skipped only when every disparity is invalid, margins
  // included.
  if (!incomplete_disparity_range) {
    return all_non_zero_elements(criteria_cost_surface);
  }

  const int rows = static_cast<int>(criteria_cost_surface.rows());
  const int cols = static_cast<int>(criteria_cost_surface.cols());
  if (rows <= 0 || cols <= 0) {
    return false;
  }

  // Clamp user range indices to surface boundaries for safety
  const int row_start = std::clamp(user_d_row_start, 0, rows - 1);
  const int row_end = std::clamp(user_d_row_end, 0, rows - 1);
  const int col_start = std::clamp(user_d_col_start, 0, cols - 1);
  const int col_end = std::clamp(user_d_col_end, 0, cols - 1);
  if (row_start > row_end || col_start > col_end) {
    return false;
  }

  // Skip if at least one disparity in user range is invalid
  return (criteria_cost_surface
              .block(row_start, col_start, row_end - row_start + 1, col_end - col_start + 1)
              .array() != 0)
      .any();
}

/**
 * @brief Get the index corresponding to the disp_val value in the disparity range
 *
 * @param disp_range disparity range
 * @param disp_val disparity value for which we want to find the index
 * @return int index of the disparity value in the disparity range
 */
int disparity_index(const P2d::VectorD& disp_range, float disp_val) {
  // std::lower_bound returns an iterator pointing to the first element
  // in the disparity_range which does not compare less than disp_val
  // std::lower_bound uses a dichotomy algorithm, that is why we use this method instead of
  // std::find
  auto it = std::lower_bound(disp_range.begin(), disp_range.end(), disp_val);
  // We measure distance between the beginning of the disparity range and the iterator returned by
  // std::lower_bound.
  return std::distance(disp_range.begin(), it);
}