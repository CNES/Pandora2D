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
This module contains functions associated to the zncc in cpp.
*/

#ifndef ZNCC_HPP
#define ZNCC_HPP

#include "operation.hpp"
#include "pandora2d_type.hpp"

const double STD_EPSILON = 1e-8;  // is 1e-16 for the variance.

/**
 * @brief Shift image according to row and columns disparities.
 * The edges are replicated at the edge of the image.
 * (These edges will not be calculated because they are invalid in the criteria data array.)
 *
 * @param image image to shift
 * @param disp_row row disparity
 * @param disp_col column disparity
 * @return P2d::Matrixf
 */
inline P2d::Matrixf shift_image(const P2d::Matrixf& image, int disp_row, int disp_col) {
  const int rows = image.rows();
  const int cols = image.cols();
  P2d::Matrixf shifted_image(rows, cols);

  int src_row;
  int src_col;

  for (int row = 0; row < rows; ++row) {
    src_row = row + disp_row;
    if (src_row < 0)
      src_row = 0;
    if (src_row >= rows)
      src_row = rows - 1;

    for (int col = 0; col < cols; ++col) {
      src_col = col + disp_col;
      if (src_col < 0)
        src_col = 0;
      if (src_col >= cols)
        src_col = cols - 1;

      shifted_image(row, col) = image(src_row, src_col);
    }
  }
  return shifted_image;
}

/**
 * @brief Compute integral image and squared integral image
 *
 * @param image image
 * @param integral_image integral image to complete
 * @param integral_image_sq integral of squared image to complete
 */
template <typename T>
void compute_integral_image(const P2d::Matrixf& image,
                            P2d::MatrixX<T>& integral_image,
                            P2d::MatrixX<T>& integral_image_sq) {
  int rows = image.rows();
  int cols = image.cols();

  integral_image.setZero(rows + 1, cols + 1);
  integral_image_sq.setZero(rows + 1, cols + 1);

  T sum_row;
  T sum_row_sq;
  T val;

  for (int row = 1; row <= rows; ++row) {
    sum_row = 0.;
    sum_row_sq = 0.;
    for (int col = 1; col <= cols; ++col) {
      // Cast to T because image type is float
      val = static_cast<T>(image(row - 1, col - 1));
      sum_row += val;
      sum_row_sq += val * val;
      integral_image(row, col) = integral_image(row - 1, col) + sum_row;
      integral_image_sq(row, col) = integral_image_sq(row - 1, col) + sum_row_sq;
    }
  }
}

/**
 * @brief Return sum of elements in a window using integral images
 *
 * If I_int is the integral image of I,
 * We have the following formula for the sum of the points contained in a window of I
 * whose upper left point is (x1, y1) and lower right point is (x2, y2):
 *
 * S = I_int(x2,y2) - I_int(x1-1,y2) - I_int(x2, y1-1) + I_int(x1-1,y1-1)
 *
 * @param integral integral image
 * @param top index of top row of the window
 * @param left index of left col of the window
 * @param bottom index of bottom row of the window
 * @param right index of right col of the window
 * @return T
 */
template <typename T>
inline T sum_window(const P2d::MatrixX<T>& integral, int top, int left, int bottom, int right) {
  return integral(bottom + 1, right + 1) - integral(top, right + 1) - integral(bottom + 1, left) +
         integral(top, left);
}

/**
 * @brief Compute integral images of:
 *
 * - shifted right image
 * - squared shifted right image
 * - product between left image and shifted right image
 *
 * @param left left image
 * @param shifted_right shifted right image
 * @param integral_right right integral image to complete
 * @param integral_right_sq squared right integral image to complete
 * @param integral_cross product of left and right integral image to complete
 */
template <typename T>
inline void compute_right_integrals(const P2d::Matrixf& left,
                                    const P2d::Matrixf& shifted_right,
                                    P2d::MatrixX<T>& integral_right,
                                    P2d::MatrixX<T>& integral_right_sq,
                                    P2d::MatrixX<T>& integral_cross) {
  const int rows = left.rows();
  const int cols = left.cols();

  // Initialize right integral images to zero matrix
  integral_right.setZero(rows + 1, cols + 1);
  integral_right_sq.setZero(rows + 1, cols + 1);
  integral_cross.setZero(rows + 1, cols + 1);

  T sum_row_right;
  T sum_row_right_sq;
  T sum_row_cross;

  T left_val;
  T right_val;

  for (int row = 1; row <= rows; ++row) {
    sum_row_right = 0.0;
    sum_row_right_sq = 0.0;
    sum_row_cross = 0.0;

    for (int col = 1; col <= cols; ++col) {
      // Cast to T because image type is float
      left_val = static_cast<T>(left(row - 1, col - 1));
      right_val = static_cast<T>(shifted_right(row - 1, col - 1));

      sum_row_right += right_val;
      sum_row_right_sq += right_val * right_val;
      sum_row_cross += left_val * right_val;

      integral_right(row, col) = integral_right(row - 1, col) + sum_row_right;
      integral_right_sq(row, col) = integral_right_sq(row - 1, col) + sum_row_right_sq;
      integral_cross(row, col) = integral_cross(row - 1, col) + sum_row_cross;
    }
  }
}

/**
 * @brief Compute ZNCC with integral images
 *
 * @param integral_left integral image from left image
 * @param integral_left_sq integral image from squared left image
 * @param integral_right integral image from right image
 * @param integral_right_sq integral image from squared right image
 * @param integral_cross integral image from product of left and right image
 * @param top_row top row of the window
 * @param left_col left column of the window
 * @param bottom_row bottom row of the window
 * @param right_col right column of the window
 * @param window_size window size
 * @return T
 */
template <typename T>
inline T calculate_zncc(const P2d::MatrixX<T>& integral_left,
                        const P2d::MatrixX<T>& integral_left_sq,
                        const P2d::MatrixX<T>& integral_right,
                        const P2d::MatrixX<T>& integral_right_sq,
                        const P2d::MatrixX<T>& integral_cross,
                        int top_row,
                        int left_col,
                        int bottom_row,
                        int right_col,
                        int window_size) {
  const int window_area = window_size * window_size;

  T sum_left = sum_window(integral_left, top_row, left_col, bottom_row, right_col);
  T sum_left_sq = sum_window(integral_left_sq, top_row, left_col, bottom_row, right_col);
  T sum_right = sum_window(integral_right, top_row, left_col, bottom_row, right_col);
  T sum_right_sq = sum_window(integral_right_sq, top_row, left_col, bottom_row, right_col);
  T sum_cross = sum_window(integral_cross, top_row, left_col, bottom_row, right_col);

  T mean_left = sum_left / window_area;
  T mean_right = sum_right / window_area;
  T var_left = sum_left_sq / window_area - mean_left * mean_left;
  T var_right = sum_right_sq / window_area - mean_right * mean_right;

  T std_left = std::sqrt(var_left);
  T std_right = std::sqrt(var_right);

  if (std_left <= STD_EPSILON || std_right <= STD_EPSILON) {
    return 0.0;
  }

  return ((sum_cross / window_area) - (mean_left * mean_right)) / (std_left * std_right);
}

#endif