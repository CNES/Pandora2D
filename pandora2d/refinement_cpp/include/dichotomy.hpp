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

#ifndef DICHOTOMY_HPP
#define DICHOTOMY_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <map>

#include "cost_volume.hpp"
#include "interpolation_filter.hpp"
#include "operation.hpp"
#include "pandora2d_type.hpp"

namespace py = pybind11;

/**
 * @brief Mapping of cost selection methods
 *
 */
const std::map<std::string, int (*)(const P2d::VectorD&)> COST_SELECTION_METHOD_MAPPING = {
    {"min", nanargmin},
    {"max", nanargmax}};

/**
 * @brief Position2D
 *
 */
struct Position2D {
  /**
   * @brief Construct a new Position 2 D object
   *
   * @param _r : Row position
   * @param _c : Column position
   */
  Position2D(unsigned int _row, unsigned int _col) : row(_row), col(_col) {};

  /**
   * @brief Construct a new Position 2 D object
   *
   */
  Position2D() : Position2D(0u, 0u) {};

  unsigned int row;  ///< Row position
  unsigned int col;  ///< Column position
};

/**
 * @brief Get the cost surfaces for one pixel
 *
 * @param cost_volume : 1D data
 * @param index : pixel index to find its cost surface
 * @param cv_size : the structure containing the dimensions of the cost volume
 * @return P2d::MatrixD of size nb_disp_row * nb_disp_col
 */
template <typename T>
P2d::MatrixD get_cost_surface(py::array_t<T>& cost_volume,
                              unsigned int index,
                              CostVolumeSize& cv_size) {
  auto index_to_position = [](unsigned int index, CostVolumeSize& cv_size) -> Position2D {
    int quot = index / (cv_size.nb_col * cv_size.nb_disps());
    int rem = index % (cv_size.nb_col * cv_size.nb_disps());
    return Position2D(quot, rem / cv_size.nb_disps());
  };

  // Recover pixel index
  Position2D p = index_to_position(index, cv_size);

  // Access to array data - 4 for cost volume dimension
  auto r_cost_volume = cost_volume.template unchecked<4>();

  // Matrix creation
  P2d::MatrixD cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);

  // Data copy
  for (std::size_t k_disp_row = 0; k_disp_row < cv_size.nb_disp_row; ++k_disp_row) {
    for (std::size_t l_disp_col = 0; l_disp_col < cv_size.nb_disp_col; ++l_disp_col) {
      cost_surface(k_disp_row, l_disp_col) = r_cost_volume(p.row, p.col, k_disp_row, l_disp_col);
    }
  }

  return cost_surface;
}

/**
 * @brief Search for the new best position
 *
 * @param cost_surface : 2D data of size nb_disp_row * nb_disp_col
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
                           std::string method_matching_cost);

/**
 * @brief Dichotomy calculation
 *
 * @param cost_volume : 1D data of size nb_row * nb_col * nb_disp_row * nb_disp_col
 * @param disparity_map_col : 1D col disparity data of size nb_row * nb_col
 * @param disparity_map_row : 1D row disparity data of size nb_row * nb_col
 * @param score_map : 1D score data of size nb_row * nb_col
 * @param criteria_map : 1D criteria data of size nb_row * nb_col
 * @param subpixel : sub-sampling of cost_volume
 * @param nb_iterations : number of iterations of the dichotomy
 * @param filter : interpolation filter
 * @param method_matching_cost : max or min
 */
template <typename T, typename U>
void compute_dichotomy(py::array_t<T> cost_volume,
                       Eigen::Ref<U> disparity_map_col,
                       Eigen::Ref<U> disparity_map_row,
                       Eigen::Ref<U> score_map,
                       U& criteria_map,
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
  CostVolumeSize cv_size = CostVolumeSize(cost_volume.shape(0), cost_volume.shape(1),
                                          cost_volume.shape(2), cost_volume.shape(3));
  auto nb_disps = cv_size.nb_disps();

  unsigned int index = -nb_disps;  //< Index on disparity_map less the first occurance
  P2d::MatrixD cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);
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
    cost_surface = get_cost_surface<T>(cost_volume, index, cv_size);
    double pos_row_disp =
        (std::is_same_v<T, float>) ? static_cast<double>(*pos_disp_row_it) : *pos_disp_row_it;
    double pos_col_disp =
        (std::is_same_v<T, float>) ? static_cast<double>(*pos_disp_col_it) : *pos_disp_col_it;
    double score = (std::is_same_v<T, float>) ? static_cast<double>(*score_it) : *score_it;

    // Loop on nb_iterations
    precision = 1. / pow(2, first_iterations);  //< Reset precision before iterations
    for (auto it = first_iterations; it <= nb_iterations; ++it) {
      search_new_best_point(cost_surface, precision, subpixel, pos_row_disp, pos_col_disp, score,
                            filter, method_matching_cost);
      // update precision
      precision /= 2.;
    }

    // update informations
    *pos_disp_row_it = (std::is_same_v<T, float>) ? static_cast<float>(pos_row_disp) : pos_row_disp;
    *pos_disp_col_it = (std::is_same_v<T, float>) ? static_cast<float>(pos_col_disp) : pos_col_disp;
    *score_it = (std::is_same_v<T, float>) ? static_cast<float>(score) : score;
  }
}

#endif