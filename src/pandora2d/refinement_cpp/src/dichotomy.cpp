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
This module contains functions associated to the Dichotomy refinement method.
*/

#include "dichotomy.hpp"
#include <array>
#include <utility>
#include "compute_cost_volumes.hpp"
#include "constant.hpp"

/**
 * @brief Search for the new best position
 *
 * @param cost_surface : 1D data of size nb_disp_row * nb_disp_col
 * @param precision : search precision (in 1/pow(2,n))
 * @param subpixel : sub-sampling of cost_volume
 * @param pos_row_disp : initial position on row
 * @param pos_col_disp : initial position on col
 * @param score : best score on cost surface (minimum or maximum)
 * @param filter : interpolation filter
 * @param method_matching_cost : max or min
 */
void search_new_best_point(const P2d::MatrixD& cost_surface,
                           const double precision,
                           const double subpixel,
                           double& pos_row_disp,
                           double& pos_col_disp,
                           double& score,
                           abstractfilter::AbstractFilter& filter,
                           std::string method_matching_cost) {
  // Used to compute new positions and new disparities based on precision
  P2d::VectorD disp_row_shifts(9);
  P2d::VectorD disp_col_shifts(9);

  disp_row_shifts << -1, -1, -1, 0, 0, 0, 1, 1, 1;
  disp_col_shifts << -1, 0, 1, -1, 0, 1, -1, 0, 1;

  // Array with the 8 new positions to be tested around the best previous point.
  P2d::VectorD new_row_pos(9);
  new_row_pos =
      disp_row_shifts * precision * subpixel + P2d::VectorD::Constant(9, pos_row_disp).eval();
  P2d::VectorD new_col_pos(9);
  new_col_pos =
      disp_col_shifts * precision * subpixel + P2d::VectorD::Constant(9, pos_col_disp).eval();

  // Interpolate points at positions (new_row_pos[i], new_col_pos[i])
  P2d::VectorD candidates(9);
  candidates = filter.interpolate(cost_surface, new_col_pos, new_row_pos, MAX_FRACTIONAL_VALUE);
  // In case a NaN is present in the kernel, candidates will be all-NaNs. Let’s restore
  // initial_position value so that best candidate search will be able to find it.
  candidates[4] = score;

  // search index of new best score
  if (not all_same(candidates)) {
    int best_index = COST_SELECTION_METHOD_MAPPING.at(method_matching_cost)(candidates);

    // Update variables
    pos_row_disp = new_row_pos[best_index];
    pos_col_disp = new_col_pos[best_index];
    score = candidates[best_index];
  }
}

/**
 * @brief Search for the new best position by interpolating the right image
 *
 * @param correlator : correlator prepared with the left window centered on the current pixel
 * @param right_image : right image
 * @param precision : search precision (1/pow(2,n), in pixels at the nth iteration)
 * @param pixel_row : row of the current pixel
 * @param pixel_col : col of the current pixel
 * @param window_size : size of the correlation window
 * @param best_row_disp : initial best row disparity, updated with the new best one
 * @param best_col_disp : initial best col disparity, updated with the new best one
 * @param best_score : initial best score, updated with the new best one (maximum)
 * @param filter : interpolation filter
 * @param right_valid_mask mask of valid pixels in the right image
 */
void search_new_best_point_on_image(const Correlator<double>& correlator,
                                    const P2d::MatrixD& right_image,
                                    double precision,
                                    int pixel_row,
                                    int pixel_col,
                                    int window_size,
                                    double& best_row_disp,
                                    double& best_col_disp,
                                    double& best_score,
                                    abstractfilter::AbstractFilter& filter,
                                    const P2d::MatrixUI& right_valid_mask) {
  // 8 neighbours of the initial best position
  constexpr std::array<std::pair<int, int>, 8> disp_shifts = {
      {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};
  // We save the initial best row and column disparities
  // because best_row_disp and best_col_disp will be updated during the search
  const double init_row_disp = best_row_disp;
  const double init_col_disp = best_col_disp;
  const int precision_inverse = static_cast<int>(1. / precision);

  // We interpolate all the correlation windows in the right image for the 8 candidates.
  // The interpolated values are stored in "grid".
  const std::optional<P2d::Matrixf> grid =
      filter.interpolate_candidate_windows(right_image, window_size, pixel_row + init_row_disp,
                                           pixel_col + init_col_disp, precision, right_valid_mask);
  if (!grid.has_value()) {
    return;
  }

  // We iterate through the candidate positions around the initial best point
  // For each candidate position, the corresponding correlation window is extracted from the
  // interpolated right image
  for (const auto& [shift_row, shift_col] : disp_shifts) {
    // Centers of the interpolated correlation windows
    const double candidate_row_disp = init_row_disp + shift_row * precision;
    const double candidate_col_disp = init_col_disp + shift_col * precision;

    // The grid is ordered differently depending on whether inverse_precision=2 or not.
    // For precision_inverse == 2, the grid is in "normal" increasing spatial order (one point every
    // half pixel). The start index is shift + 1 and the step is precision_inverse. For example, for
    // shift={-1,0}, the start index would be 0 for row and 1 for column and we must use a
    // precision_inverse step from that point to find the other points in the window.

    // For precision_inverse != 2, the grid is ordered "by block".
    // Each candidate window is stored contiguously in its own window_size x window_size block,
    // the top-left block corresponds to shift {-1, -1} and the bottom-right block to shift {1, 1})
    // the start index is (shift + 1) * window_size and the step is 1.
    // For example, for shift={-1,0}, precision_inverse=4 and window_size = 3 :
    // the start index would be 0 for row and 3 for column,
    // and we must use a step of 1 to find the other points in the window.
    const int grid_start_row =
        (precision_inverse == 2) ? (shift_row + 1) : (shift_row + 1) * window_size;
    const int grid_start_col =
        (precision_inverse == 2) ? (shift_col + 1) : (shift_col + 1) * window_size;
    const int grid_step = (precision_inverse == 2) ? precision_inverse : 1;

    // We extract the correlation window from the interpolated right image for the current candidate
    P2d::Matrixf right_window(window_size, window_size);
    for (int row = 0; row < window_size; ++row) {
      for (int col = 0; col < window_size; ++col) {
        right_window(row, col) =
            (*grid)(grid_start_row + row * grid_step, grid_start_col + col * grid_step);
      }
    }
    // NaN values would make the histograms of the mutual information undefined
    if (right_window.hasNaN())
      continue;
    // We compute correlation between the left window and the right window of the current candidate
    auto compute_score = [&](const auto& correlator_impl) -> double {
      return correlator_impl(right_window);
    };
    const double candidate_score = std::visit(compute_score, correlator);
    // Update the best score and disparities if the current candidate has a better score
    if (candidate_score > best_score) {
      best_score = candidate_score;
      best_row_disp = candidate_row_disp;
      best_col_disp = candidate_col_disp;
    }
  }
}

/**
 * @brief Create and prepare a correlator with the given left window
 *
 * @param left_window left window to prepare
 * @param method_matching_cost matching cost method
 * @return Correlator<double>
 */
Correlator<double> make_prepared_left_window(const P2d::Matrixf& left_window,
                                             const std::string& method_matching_cost) {
  // zncc-optim-1 relies on integral images, which are useless on a single window
  const std::string correlator_method =
      (method_matching_cost == "zncc" || method_matching_cost == "zncc-optim-1")
          ? "zncc-optim-2"
          : method_matching_cost;

  Correlator<double> correlator = make_correlator<double>(correlator_method);
  // clang-format off
  std::visit(
      [&](auto& correlator_impl) {
        correlator_impl.prepare_left_window(left_window);
      },
      correlator);
  // clang-format on
  return correlator;
}

/**
 * @brief Resample the right image to refine the disparity estimation at a given position.
 *
 * @param left_image left image
 * @param right_image_double right image in double precision
 * @param pixel_row row of the current pixel
 * @param pixel_col column of the current pixel
 * @param window_size size of the correlation window
 * @param nb_iterations number of iterations of the dichotomy
 * @param first_iteration index of the first dichotomy iteration
 * @param filter interpolation filter
 * @param method_matching_cost matching cost method
 * @param subpixel sub-sampling of the cost volume
 * @param disp_row_min minimum row disparity
 * @param disp_col_min minimum column disparity
 * @param pos_row_disp position of the initial best row disparity
 * @param pos_col_disp position of the initial best column disparity
 * @param score matching cost score at the initial best disparity position
 * @param right_valid_mask mask of valid pixels in the right image
 */
void image_resampling(const P2d::Matrixf& left_image,
                      const P2d::MatrixD& right_image_double,
                      int pixel_row,
                      int pixel_col,
                      int window_size,
                      int nb_iterations,
                      int first_iteration,
                      abstractfilter::AbstractFilter& filter,
                      const std::string& method_matching_cost,
                      int subpixel,
                      double disp_row_min,
                      double disp_col_min,
                      double& pos_row_disp,
                      double& pos_col_disp,
                      double& score,
                      const P2d::MatrixUI& right_valid_mask) {
  // pos_row_disp and pos_col_disp correspond to the indices of the best initial disparities
  // We must convert these indices to disparity values in order to locate the central pixel in the
  // window of the right image that we want to interpolate.
  double init_best_row_disp = disp_row_min + pos_row_disp / subpixel;
  double init_best_col_disp = disp_col_min + pos_col_disp / subpixel;

  // Extract the left window centered on the current pixel to prepare the correlator.
  P2d::Matrixf left_window = get_window(left_image, window_size, pixel_row, pixel_col);
  const Correlator<double> correlator =
      make_prepared_left_window(left_window, method_matching_cost);

  double precision = 1. / pow(2, first_iteration);

  // Dichotomy loop: at each iteration, search around the current best disparities
  // with the current precision, and keep the candidate with the best score.
  // init_best_row_disp, init_best_col_disp and score are updated in place.
  for (auto it = first_iteration; it <= nb_iterations; ++it) {
    search_new_best_point_on_image(correlator, right_image_double, precision, pixel_row, pixel_col,
                                   window_size, init_best_row_disp, init_best_col_disp, score,
                                   filter, right_valid_mask);
    // Update precision
    precision /= 2.;
  }

  // Convert the refined disparity values back to indices (output values)
  pos_row_disp = (init_best_row_disp - disp_row_min) * subpixel;
  pos_col_disp = (init_best_col_disp - disp_col_min) * subpixel;
}
