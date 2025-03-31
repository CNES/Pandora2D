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
This module contains functions associated to the Dichotomy refinement method.
*/

#include "dichotomy.hpp"
#include "constant.hpp"

namespace py = pybind11;

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
