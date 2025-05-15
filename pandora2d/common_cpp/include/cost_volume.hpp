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
This module contains cost volume struct.
*/

#ifndef COMMON_COST_VOLUME_HPP
#define COMMON_COST_VOLUME_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pandora2d_type.hpp"

namespace py = pybind11;

/**
 * @brief Cost volume size
 *
 * Structure containing the 4D size of a cost volume
 */
struct CostVolumeSize {
  /**
   * @brief Construct a new CostVolumeSize object
   *
   * @param _r : Number of rows on cost_volume element
   * @param _c : Number of columns on cost_volume element
   * @param _dr : Number of row disparities
   * @param _dc : Number of col disparities
   */
  CostVolumeSize(unsigned int _r, unsigned int _c, unsigned int _dr, unsigned int _dc)
      : nb_row(_r), nb_col(_c), nb_disp_row(_dr), nb_disp_col(_dc) {};

  /**
   * @brief Construct a new CostVolumeSize object
   *
   */
  CostVolumeSize() : CostVolumeSize(0u, 0u, 0u, 0u) {};

  /**
   * @brief Construct a new CostVolumeSize object
   *
   * @param cv_size : Eigen vector with cost_volume size informations
   */
  CostVolumeSize(P2d::VectorD& cv_size)
      : CostVolumeSize(cv_size[0], cv_size[1], cv_size[2], cv_size[3]) {};

  /**
   * @brief Construct a new CostVolumeSize object
   *
   * @param cv_size : std::vector with cost_volume size informations
   */
  CostVolumeSize(std::vector<size_t>& cv_size)
      : CostVolumeSize(cv_size[0], cv_size[1], cv_size[2], cv_size[3]) {};

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
 * @return P2d::MatrixU of size nb_disp_row * nb_disp_col
 * (U can be either a float, a double or an uint_8)
 */
template <typename T, typename U>
Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> get_cost_surface(const py::array_t<T>& cost_volume,
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
  Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> cost_surface(cv_size.nb_disp_row,
                                                                cv_size.nb_disp_col);

  // Data copy
  for (std::size_t k_disp_row = 0; k_disp_row < cv_size.nb_disp_row; ++k_disp_row) {
    for (std::size_t l_disp_col = 0; l_disp_col < cv_size.nb_disp_col; ++l_disp_col) {
      cost_surface(k_disp_row, l_disp_col) = r_cost_volume(p.row, p.col, k_disp_row, l_disp_col);
    }
  }

  return cost_surface;
}

#endif