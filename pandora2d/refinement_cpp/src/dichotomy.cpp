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
This module contains functions associated to the Dichotomy refinement method.
*/

#include "dichotomy.hpp"
#include "alias.hpp"

/**
 * @brief Function to find the index of the minimum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmin(const Eigen::VectorXd& vec) {
  int min_index = -1;
  double min_value = std::numeric_limits<double>::infinity();
  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isnan(vec[i]) && vec[i] < min_value) {
      min_value = vec[i];
      min_index = i;
    }
  }
  return min_index;
}

/**
 * @brief Function to find the index of the maximum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmax(const Eigen::VectorXd& vec) {
  int max_index = -1;
  double max_value = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isnan(vec[i]) && vec[i] > max_value) {
      max_value = vec[i];
      max_index = i;
    }
  }
  return max_index;
}

/**
 * @brief Check if all the elements are the same
 *
 * @param data
 * @return true : all elements are the same
 * @return false : not all elements are the same
 */
bool all_same(const Eigen::VectorXd& data) {
  auto value_tested = *(data.begin());
  for (auto d : data) {
    if (d != value_tested)
      return false;
  }
  return true;
}

/**
 * @brief Get the cost surfaces for one pixel
 *
 * @brief Get the cost surfaces for one pixel
 *
 * @param cost_volume : 1D data
 * @param index : pixel index to find its cost surface
 * @param cv_size : the structure containing the dimensions of the cost volume
 * @return EEigen::MatrixXd of size nb_disp_row * nb_disp_col
 */
Eigen::MatrixXd get_cost_surface(const Eigen::VectorXd& cost_volume,
                                 unsigned int index,
                                 Cost_volume_size& cv_size) {
  return cost_volume.segment(index, cv_size.nb_disps())
      .reshaped<Eigen::RowMajor>(cv_size.nb_disp_row, cv_size.nb_disp_col);
}

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
void search_new_best_point(const Eigen::MatrixXd& cost_surface,
                           const double precision,
                           const double subpixel,
                           double& pos_row_disp,
                           double& pos_col_disp,
                           double& score,
                           abstractfilter::AbstractFilter& filter,
                           std::string method_matching_cost) {
  // Used to compute new positions and new disparities based on precision
  Eigen::VectorXd disp_row_shifts(9);
  Eigen::VectorXd disp_col_shifts(9);

  disp_row_shifts << -1, -1, -1, 0, 0, 0, 1, 1, 1;
  disp_col_shifts << -1, 0, 1, -1, 0, 1, -1, 0, 1;

  // Array with the 8 new positions to be tested around the best previous point.
  Eigen::VectorXd new_row_pos(9);
  new_row_pos =
      disp_row_shifts * precision * subpixel + Eigen::VectorXd::Constant(9, pos_row_disp).eval();
  Eigen::VectorXd new_col_pos(9);
  new_col_pos =
      disp_col_shifts * precision * subpixel + Eigen::VectorXd::Constant(9, pos_col_disp).eval();

  // Interpolate points at positions (new_row_pos[i], new_col_pos[i])
  Eigen::VectorXd candidates(9);
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
 * @brief Dichotomy calculation
 *
 * @param cost_volume : 1D data of size nb_row * nb_col * nb_disp_row * nb_disp_col
 * @param disparity_map_col : 1D col disparity data of size nb_row * nb_col
 * @param disparity_map_row : 1D row disparity data of size nb_row * nb_col
 * @param score_map : 1D score data of size nb_row * nb_col
 * @param criteria_map : 1D criteria data of size nb_row * nb_col
 * @param cv_size : cost volume size information
 * @param subpixel :sub-sampling of cost_volume
 * @param nb_iterations : number of iterations of the dichotomy
 * @param filter : interpolation filter
 * @param method_matching_cost : max or min
 */
void compute_dichotomy(Eigen::VectorXd& cost_volume,
                       Eigen::Ref<Eigen::VectorXd> disparity_map_col,
                       Eigen::Ref<Eigen::VectorXd> disparity_map_row,
                       Eigen::Ref<Eigen::VectorXd> score_map,
                       Eigen::VectorXd& criteria_map,
                       Cost_volume_size& cv_size,
                       int subpixel,
                       int nb_iterations,
                       abstractfilter::AbstractFilter& filter,
                       std::string method_matching_cost) {
  // Get parameters
  auto first_iterations = (subpixel != 4) ? subpixel : 3;  //< subpixel is 1, 2 or 4
  auto pos_disp_col_it = disparity_map_col.begin();
  auto pos_disp_row_it = disparity_map_row.begin();
  auto score_it = score_map.begin();
  auto crit_it = criteria_map.begin();
  auto nb_disps = cv_size.nb_disps();

  unsigned int index = -nb_disps;  //< Index on disparity_map less the first occurance
  Eigen::MatrixXd cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);
  double precision = 0.;

  // Loop on each image point calculated
  for (; pos_disp_col_it != disparity_map_col.end();
       ++pos_disp_col_it, ++pos_disp_row_it, ++score_it, ++crit_it) {
    // update index
    index += nb_disps;

    // taking into account the peak at the edge & invalid disparities (== 1 in the array)
    if (*crit_it == 1.)
      continue;

    // Check initial disparity is not nan
    if (std::isnan(*pos_disp_row_it) or std::isnan(*pos_disp_col_it))
      continue;

    // get cost_surface
    cost_surface = get_cost_surface(cost_volume, index, cv_size);

    // Loop on nb_iterations
    precision = 1. / pow(2, first_iterations);  //< Reset precision before iterations
    for (auto it = first_iterations; it <= nb_iterations; ++it) {
      search_new_best_point(cost_surface, precision, subpixel, *pos_disp_row_it, *pos_disp_col_it,
                            *score_it, filter, method_matching_cost);
      // update precision
      precision /= 2.;
    }
  }
}