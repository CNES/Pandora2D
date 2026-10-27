/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
 * Copyright (c) 2026 Institut Français de Recherche pour l'Exploitation de la MER (IFREMER).
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
This module contains functions associated to the Abstract filter class for cpp.
*/

#include "interpolation_filter.hpp"
#include <cmath>
#include <limits>

namespace abstractfilter {

// Constructor
AbstractFilter::AbstractFilter(int size, Margins margins) : m_size(size), m_margins(margins) {}

P2d::VectorD AbstractFilter::get_coeffs(const double fractional_shift) {
  return P2d::VectorD();
}

// Apply
double AbstractFilter::apply(const P2d::MatrixD& resampling_area,
                             const P2d::VectorD& row_coeff,
                             const P2d::VectorD& col_coeff) const {
  P2d::VectorD intermediate_result = resampling_area * col_coeff;
  return row_coeff.dot(intermediate_result);
}

// Interpolate
P2d::VectorD AbstractFilter::interpolate(const P2d::MatrixD& image,
                                         const P2d::VectorD& col_positions,
                                         const P2d::VectorD& row_positions,
                                         const double max_fractional_value) {
  // Initialisation of the result list
  P2d::VectorD interpolated_positions = P2d::VectorD::Zero(col_positions.size());

  // AbstractFilter
  const Margins& my_margins = AbstractFilter::m_margins;
  const int filter_size = AbstractFilter::m_size;

  auto col_it = col_positions.begin();
  auto row_it = row_positions.begin();
  auto result_it = interpolated_positions.begin();

  for (; col_it != col_positions.end(); ++col_it, ++row_it, ++result_it) {
    // get_coeffs method receives positive coefficients
    double fractional_row = std::abs(*row_it - std::floor(*row_it));
    double fractional_col = std::abs(*col_it - std::floor(*col_it));

    // If the subpixel shift is too close to 1, max_fractional_value is returned
    // to avoid rounding.
    if (1 - fractional_row < EPSILON) {
      fractional_row = max_fractional_value;
    }
    if (1 - fractional_col < EPSILON) {
      fractional_col = max_fractional_value;
    }

    // Get interpolation coefficients for fractional_row and fractional_col shifts
    P2d::VectorD coeffs_row = this->get_coeffs(fractional_row);
    P2d::VectorD coeffs_col = this->get_coeffs(fractional_col);

    /*
    Computation of the top left point of the resampling area
    on which the interpolating coefficients will be applied with apply method
    In cost_surface, row dimension is disp_col and column dimension is disp_row,
    then we use margins.left for row and margins.up for col
    */
    int top_left_area_row = *row_it - my_margins.left;
    int top_left_area_col = *col_it - my_margins.up;

    // Resampling area to which we will apply the interpolator coefficients
    P2d::MatrixD resampling_area =
        image.block(top_left_area_row, top_left_area_col, filter_size, filter_size);

    // Application of the interpolator coefficients on resampling area
    const auto result = apply(resampling_area, coeffs_row, coeffs_col);
    *result_it = result;
  }

  return interpolated_positions;
}

// Build the mapping from the compact grid to the window pixels
Eigen::VectorXi AbstractFilter::build_grid_to_window_mapping(int window_size,
                                                             int inverse_precision) const {
  // We construct the idx vector to establish a correspondence between the positions in the grid and
  // the positions in the right image.
  // We have: k: the index in the grid where the interpolated values are stored
  // and idx[k]: the position in the corresponding right image,
  // starting from the first point to be interpolated (first_row or first_col).
  // Only the points contained in one of the correlation windows for the 8 candidates are present in
  // the grid.
  if (inverse_precision == 2) {
    // For window_size = 3 and inverse_precision = 2, idx = [0, 1, 2, 3, 4, 5, 6]
    // For origin_row = 3, we obtain first_row = 1.5 and image row positions equal to
    // [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    return Eigen::VectorXi::LinSpaced(2 * window_size + 1, 0, 2 * window_size);
  }

  // We reserve the size in advance for idx.
  Eigen::VectorXi idx(3 * window_size);

  // For inverse_precision != 2 the grid is ordered "by block".
  // Each candidate window is stored contiguously in its own window_size x window_size block.
  // Thus, for window_size = 3 and inverse_precision = 4, the second point (index 1 in row or
  // column) to be interpolated in the right image corresponds to index 4 in the grid. Then we have
  // : idx = [0, 4, 8, 1, 5, 9, 2, 6, 10] For origin_row = 3, we obtain first row = 1.75 and image
  // row positions equal to [1.75, 2.75, 3.75, 2., 3., 4., 2.25, 3.25, 4.25]

  idx << Eigen::VectorXi::LinSpaced(window_size, 0, (window_size - 1) * inverse_precision),
      Eigen::VectorXi::LinSpaced(window_size, 1, 1 + (window_size - 1) * inverse_precision),
      Eigen::VectorXi::LinSpaced(window_size, 2, 2 + (window_size - 1) * inverse_precision);

  return idx;
}

// Build the candidate window positions for interpolation
P2d::VectorD AbstractFilter::build_candidate_window_positions(const Eigen::VectorXi& indices,
                                                              double origin,
                                                              double precision,
                                                              int half_window) const {
  // We calculate the position of the top-left corner of the grid that will store the interpolated
  // values. We subtract a half window to cover the entire window, then subtract another precision
  // to cover the candidate shift = -1.
  const double first_position = origin - half_window - precision;

  // We calculate the correspondences between the positions in the right image
  // (starting from first row and first col) and the grid indices.
  return indices.cast<double>() * precision +
         P2d::VectorD::Constant(indices.size(), first_position);
}

// Interpolate candidate windows
std::optional<P2d::Matrixf> AbstractFilter::interpolate_candidate_windows(const P2d::MatrixD& image,
                                                                          int window_size,
                                                                          double origin_row,
                                                                          double origin_col,
                                                                          double precision) {
  const Margins& margins = m_margins;
  const int filter_size = m_size;
  const int half_window = window_size / 2;
  const int inverse_precision = static_cast<int>(1. / precision);
  // Build the mapping from the compact grid which will store the interpolated values
  // to the window pixels of the 8 candidate positions around the current best point
  const Eigen::VectorXi idx = build_grid_to_window_mapping(window_size, inverse_precision);
  const int axis_size = static_cast<int>(idx.size());

  const P2d::VectorD candidate_image_rows =
      build_candidate_window_positions(idx, origin_row, precision, half_window);
  const P2d::VectorD candidate_image_cols =
      build_candidate_window_positions(idx, origin_col, precision, half_window);

  // We initialize the grid to NaN. The points that will be skipped will remain NaN.
  P2d::Matrixf grid =
      P2d::Matrixf::Constant(axis_size, axis_size, std::numeric_limits<float>::quiet_NaN());

  const int max_positions = axis_size * axis_size;
  P2d::VectorD interpolation_rows(max_positions);
  P2d::VectorD interpolation_cols(max_positions);
  Eigen::VectorXi grid_rows(max_positions);
  Eigen::VectorXi grid_cols(max_positions);

  int nb_positions = 0;

  // We iterate through the grid and keep only the points that must be interpolated.
  // For each kept point, we store its position in the right image and its row and column in the
  // grid.
  for (int axis_row = 0; axis_row < axis_size; ++axis_row) {
    // We skip the rows for which the filter area would go outside the right image
    const int top_left_area_row = candidate_image_rows[axis_row] - margins.left;
    if (top_left_area_row < 0 || top_left_area_row + filter_size > image.rows()) {
      continue;
    }
    for (int axis_col = 0; axis_col < axis_size; ++axis_col) {
      // The points of the correlation window of the the initial best candidate (shift 0, 0)
      // verify idx % inverse_precision == 1 on both axes.
      // The initial best candidate score is already known, so we skip its window.
      if (idx[axis_row] % inverse_precision == 1 && idx[axis_col] % inverse_precision == 1) {
        continue;
      }
      // We skip the columns for which the filter area would go outside the right image
      const int top_left_area_col = candidate_image_cols[axis_col] - margins.up;
      if (top_left_area_col < 0 || top_left_area_col + filter_size > image.cols()) {
        continue;
      }
      // interpolation_rows and interpolation_cols are the list of points sent to interpolate:
      // one entry per kept point, so each value of candidate_image_rows is repeated once per kept
      // column. grid_rows and grid_cols store the row and column of the grid where the result will
      // be written. For inverse_precision = 2, window_size = 3 and origin_row = 3,
      // candidate_image_rows = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
      // andconsidering the filter area fits in the right image,
      // Row 0 (1.5) keeps its 7 columns, so 1.5 is written 7 times.
      // Row 1 (2.0) loses columns 1, 3 and 5 (window of the best candidate), so 2.0 is written 4
      // times.
      interpolation_rows[nb_positions] = candidate_image_rows[axis_row];
      interpolation_cols[nb_positions] = candidate_image_cols[axis_col];
      grid_rows[nb_positions] = axis_row;
      grid_cols[nb_positions] = axis_col;
      ++nb_positions;
    }
  }

  if (nb_positions == 0) {
    return std::nullopt;
  }

  // Vectors allocated to axis_size² (max_positions) at initialization, but borders and
  // best candidate window skip indices, leaving uninitialized tail from nb_positions to
  // max_positions. conservativeResize(nb_positions) trims the uninitialized tail while
  // preserving computed values, since interpolate() would process the full allocation size.
  interpolation_rows.conservativeResize(nb_positions);
  interpolation_cols.conservativeResize(nb_positions);

  const P2d::VectorD interpolated_values =
      interpolate(image, interpolation_cols, interpolation_rows, MAX_FRACTIONAL_VALUE);
  // interpolate keeps the order of the points, so the i-th interpolated value is written
  // at the i-th stored row and column of the grid.
  // The points that were skipped remain NaN.
  // For inverse_precision = 2 and window_size = 3, the 9 points with row and column in {1, 3, 5}
  // stay NaN. For inverse_precision = 4 and window_size = 3, the central block (rows 3 to 5,
  // columns 3 to 5) stays NaN.
  for (int index = 0; index < nb_positions; ++index) {
    grid(grid_rows[index], grid_cols[index]) = static_cast<float>(interpolated_values[index]);
  }

  return grid;
}

}  // namespace abstractfilter
