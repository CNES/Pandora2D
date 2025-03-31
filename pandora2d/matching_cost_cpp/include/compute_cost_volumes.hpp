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
This module contains functions associated to the computation of cost volumes in cpp.
*/

#ifndef COMPUTE_COST_VOLUMES_HPP
#define COMPUTE_COST_VOLUMES_HPP

#include <Eigen/Dense>

#include "cost_volume.hpp"
#include "mutual_information.hpp"

/**
 * @brief Get the matching cost window
 *
 * @param img image
 * @param window_size size of the matching cost window
 * @param index_row row index of the center of the window
 * @param index_col col index of the center of the window
 * @return P2d::MatrixD
 */
P2d::MatrixD get_window(const P2d::MatrixD& img, int window_size, int index_row, int index_col);

/**
 * @brief Get the index corresponding to the correct interpolated right image
 * according to subpix value
 *
 * @param subpix value
 * @param disp_row value
 * @param disp_col value
 * @return int right index
 */
int interpolated_right_image_index(int subpix, double disp_row, double disp_col);

/**
 * @brief Check if en Eigen matrix contains an element given as parameter
 *
 * @param matrix Eigen matrix
 * @param element to check
 * @return true
 * @return false
 */
bool contains_element(const P2d::MatrixD& matrix, double element);

/**
 * @brief Compute the cost values
 *
 * @param left image
 * @param right list of right images
 * @param cv_values initialized cost values
 * @param cv_size : cost volume size information
 * @param disp_range_row cost volumes row disparity range
 * @param disp_range_col cost volumes col disparity range
 * @param offset_cv_img_row row offset between first index of cv and image (ROI case)
 * @param offset_cv_img_col col offset between first index of cv and image (ROI case)
 * @param window_size size of the correlation window
 * @param step [step_row, step_col]
 * @param no_data no data value in img
 *
 * @return P2d::VectorD computed cost values
 */
void compute_cost_volumes_cpp(const P2d::MatrixD& left,
                              const std::vector<P2d::MatrixD>& right,
                              Eigen::Ref<P2d::VectorD> cv_values,
                              CostVolumeSize& cv_size,
                              const P2d::VectorD& disp_range_row,
                              const P2d::VectorD& disp_range_col,
                              int offset_cv_img_row,
                              int offset_cv_img_col,
                              int window_size,
                              const Eigen::Vector2i& step,
                              const double no_data);

#endif