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

#include "pandora2d_type.hpp"

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

#endif