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

#ifndef DICHOTOMY_HPP
#define DICHOTOMY_HPP

#include <Eigen/Dense>
#include <map>

#include "interpolation_filter.hpp"

/**
 * @brief Function to find the index of the minimum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmin(const Eigen::VectorXd& vec);

/**
 * @brief Function to find the index of the maximum element, ignoring NaNs
 *
 * @param vec : data in the eigen vector type
 * @return int : return index (first element if all elements are the same)
 */
int nanargmax(const Eigen::VectorXd& vec);

/**
 * @brief Mapping of cost selection methods
 *
 */
const std::map<std::string, int (*)(const Eigen::VectorXd&)> COST_SELECTION_METHOD_MAPPING = {
    {"min", nanargmin},
    {"max", nanargmax}};

/**
 * @brief Check if all the elements are the same
 *
 * @param data
 * @return true : all elements are the same
 * @return false : not all elements are the same
 */
bool all_same(const Eigen::VectorXd& data);

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
 * @brief Cost volume size
 *
 * Structure containing the 4D size of a cost volume
 */
struct Cost_volume_size {
  /**
   * @brief Construct a new Cost_volume_size object
   *
   * @param _r : Number of rows on cost_volume element
   * @param _c : Number of columns on cost_volume element
   * @param _dr : Number of row disparities
   * @param _dc : Number of col disparities
   */
  Cost_volume_size(unsigned int _r, unsigned int _c, unsigned int _dr, unsigned int _dc)
      : nb_row(_r), nb_col(_c), nb_disp_row(_dr), nb_disp_col(_dc) {};

  /**
   * @brief Construct a new Cost_volume_size object
   *
   */
  Cost_volume_size() : Cost_volume_size(0u, 0u, 0u, 0u) {};

  /**
   * @brief Construct a new Cost_volume_size object
   *
   * @param cv_size : Vector with cost_volume size informations
   */
  Cost_volume_size(Eigen::VectorXd& cv_size)
      : Cost_volume_size(cv_size[0], cv_size[1], cv_size[2], cv_size[3]) {};

  /**
   * @brief Get the cost volume size : nb_row * nb_col * nb_disp_row * nb_disp_col
   *
   * @return unsigned int
   */
  unsigned int size() { return nb_row * nb_col * nb_disp_row * nb_disp_col; };

  /**
   * @brief Get the disparity number : nb_disp_row * nb_disp_col
   *
   * @return unsigned int
   */
  unsigned int nb_disps() { return nb_disp_row * nb_disp_col; };

  unsigned int nb_row;       ///< Number of rows on cost_volume element
  unsigned int nb_col;       ///< Number of columns on cost_volume element
  unsigned int nb_disp_row;  ///< Number of row disparities
  unsigned int nb_disp_col;  ///< Number of col disparities
};

/**
 * @brief Get the cost surfaces for one pixel
 *
 * @param cost_volume : 1D data
 * @param index : pixel index to find its cost surface
 * @param cv_size : the structure containing the dimensions of the cost volume
 * @return Eigen::MatrixXd of size nb_disp_row * nb_disp_col
 */
Eigen::MatrixXd get_cost_surface(const Eigen::VectorXd& cost_volume,
                                 unsigned int index,
                                 Cost_volume_size& cv_size);

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
void search_new_best_point(const Eigen::MatrixXd& cost_surface,
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
 * @param cv_size : cost volume size information
 * @param subpixel : sub-sampling of cost_volume
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
                       std::string method_matching_cost);

#endif