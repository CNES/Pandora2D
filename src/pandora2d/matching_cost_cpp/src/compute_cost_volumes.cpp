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
P2d::Matrixf get_window(const P2d::Matrixf& image,
			int window_size,
			int index_row,
			int index_col) {
  const int offset = window_size / 2;

  // Get first row and column of the window
  int start_row = std::max(0, index_row - offset);
  int start_col = std::max(0, index_col - offset);

  // Get last row and column of the window
  int nb_rows_img = image.rows();
  int nb_cols_img = image.cols();
  int end_row = std::min(nb_rows_img - 1, index_row + offset);
  int end_col = std::min(nb_cols_img - 1, index_col + offset);

  // if the window is out of the image,
  // nb_rows_window or nb_cols_window are < 0
  // in this case we return an empty window
  int nb_rows_window = std::max(0, end_row - start_row + 1);
  int nb_cols_window = std::max(0, end_col - start_col + 1);

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
