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

#ifndef DICHOTOMY_HPP
#define DICHOTOMY_HPP

#include <Eigen/Dense>
#include <map>

#include "compute_cost_volumes.hpp"
#include "cost_volume.hpp"
#include "interpolation_filter.hpp"
#include "operation.hpp"
#include "pandora2d_type.hpp"

/**
 * @brief Mapping of cost selection methods
 *
 */
const std::map<std::string, int (*)(const P2d::VectorD&)> COST_SELECTION_METHOD_MAPPING = {
    {"min", nanargmin},
    {"max", nanargmax}};

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
 * @param type_method_matching_cost : max or min
 */
void search_new_best_point(const P2d::MatrixD& cost_surface,
                           const double precision,
                           const double subpixel,
                           double& pos_row_disp,
                           double& pos_col_disp,
                           double& score,
                           abstractfilter::AbstractFilter& filter,
                           std::string type_method_matching_cost);

/**
 * @brief Search for the new best position by interpolating the right image
 *
 * The 8 neighbours of the current best disparity, shifted by precision, are tested. Their right
 * windows are interpolated in one call as the union of the 8 grids, then each window is extracted
 * and correlated with the left window. Neighbours whose extracted window contains NaN (window or
 * filter margins outside the right image) are ignored.
 *
 * @param correlator : correlator prepared with the left window centered on the current pixel
 * @param right_image : right image
 * @param precision : search precision (1/pow(2,n), in pixels, at the nth iteration)
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
                                    const P2d::MatrixUI& right_valid_mask);

/**
 * @brief Create and prepare a correlator with the given left window
 *
 * @param left_window left window to prepare
 * @param method_matching_cost matching cost method
 * @return Correlator<double>
 */
Correlator<double> make_prepared_left_window(const P2d::Matrixf& left_window,
                                             const std::string& method_matching_cost);

/**
 * @brief Resample the right image to refine the disparity estimation
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
                      const P2d::MatrixUI& right_valid_mask);

/**
 * @brief Resample the cost surface to refine the disparity estimation
 *
 * @param cost_volume 1D data of size nb_row * nb_col * nb_disp_row * nb_disp_col
 * @param index_disp pixel index used to find its cost surface
 * @param cv_size cost volumes size
 * @param subpixel sub-sampling of cost_volume
 * @param nb_iterations number of iterations of the dichotomy
 * @param first_iteration index of the first dichotomy iteration
 * @param filter interpolation filter
 * @param type_method_matching_cost type of matching cost (max or min)
 * @param pos_row_disp position of initial row disparity
 * @param pos_col_disp position of initial column disparity
 * @param score matching cost score at the initial disparity position
 */
template <typename T>
void cost_surface_resampling(py::array_t<T>& cost_volume,
                             std::size_t index_disp,
                             const CostVolumeSize& cv_size,
                             int subpixel,
                             int nb_iterations,
                             int first_iteration,
                             abstractfilter::AbstractFilter& filter,
                             const std::string& type_method_matching_cost,
                             double& pos_row_disp,
                             double& pos_col_disp,
                             double& score) {
  P2d::MatrixD cost_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);
  cost_surface = get_cost_surface<T, double>(cost_volume, index_disp, cv_size);

  double precision = 1. / pow(2, first_iteration);
  for (auto it = first_iteration; it <= nb_iterations; ++it) {
    search_new_best_point(cost_surface, precision, subpixel, pos_row_disp, pos_col_disp, score,
                          filter, type_method_matching_cost);
    // update precision
    precision /= 2.;
  }
}

/**
 * @brief Refine disparities by resampling the cost surface
 *
 * @param cost_volume : 1D data of size nb_row * nb_col * nb_disp_row * nb_disp_col
 * @param disparity_map_col : 1D col disparity data of size nb_row * nb_col
 * @param disparity_map_row : 1D row disparity data of size nb_row * nb_col
 * @param score_map : 1D score data of size nb_row * nb_col
 * @param invalid_map : 1D invalid (peak on edge and invalid) data of size nb_row * nb_col
 * @param criteria_map : 1D criteria data of nb_row * nb_col * nb_disp_row * nb_disp_col
 * @param subpixel : sub-sampling of cost_volume
 * @param nb_iterations : number of iterations of the dichotomy
 * @param filter : interpolation filter
 * @param type_method_matching_cost : max or min
 */
template <typename T, typename U>
void compute_dichotomy_cost_surface(py::array_t<T> cost_volume,
                                    Eigen::Ref<U> disparity_map_col,
                                    Eigen::Ref<U> disparity_map_row,
                                    Eigen::Ref<U> score_map,
                                    const Eigen::Ref<const P2d::VectorX<uint8_t>>& invalid_map,
                                    py::array_t<T> criteria_map,
                                    int subpixel,
                                    int nb_iterations,
                                    abstractfilter::AbstractFilter& filter,
                                    std::string type_method_matching_cost) {
  // Get parameters
  auto first_iteration = (subpixel != 4) ? subpixel : 3;  //< subpixel is 1, 2 or 4
  auto pos_disp_col_it = disparity_map_col.begin();
  auto pos_disp_row_it = disparity_map_row.begin();
  auto score_it = score_map.begin();
  auto invalid_it = invalid_map.begin();
  CostVolumeSize cv_size = CostVolumeSize(cost_volume.shape(0), cost_volume.shape(1),
                                          cost_volume.shape(2), cost_volume.shape(3));
  auto nb_disps = cv_size.nb_disps();
  P2d::MatrixUI criteria_surface(cv_size.nb_disp_row, cv_size.nb_disp_col);
  auto margins = filter.get_margins();
  auto filter_size = filter.get_size();

  // Loop on each image calculated point
  std::size_t index = 0;
  for (; pos_disp_col_it != disparity_map_col.end();
       ++pos_disp_col_it, ++pos_disp_row_it, ++score_it, ++invalid_it, ++index) {
    // Taking into account the peak at the edge & invalid disparities (== 1 in the array)
    if (*invalid_it == 1)
      continue;
    // Check initial disparity is not nan
    if (std::isnan(*pos_disp_row_it) or std::isnan(*pos_disp_col_it))
      continue;

    double pos_row_disp =
        (std::is_same_v<T, float>) ? static_cast<double>(*pos_disp_row_it) : *pos_disp_row_it;
    double pos_col_disp =
        (std::is_same_v<T, float>) ? static_cast<double>(*pos_disp_col_it) : *pos_disp_col_it;
    double score = (std::is_same_v<T, float>) ? static_cast<double>(*score_it) : *score_it;

    const std::size_t index_disp = index * nb_disps;
    // Check filter area
    criteria_surface = get_cost_surface<T, uint8_t>(criteria_map, index_disp, cv_size);
    const int top_left_area_row = *pos_disp_row_it - margins.left;
    const int top_left_area_col = *pos_disp_col_it - margins.up;
    const P2d::MatrixUI filter_area =
        criteria_surface.block(top_left_area_row, top_left_area_col, filter_size, filter_size);
    if (not filter_area.isZero())
      continue;
    // Compute resampling on cost surface
    cost_surface_resampling<T>(cost_volume, index_disp, cv_size, subpixel, nb_iterations,
                               first_iteration, filter, type_method_matching_cost, pos_row_disp,
                               pos_col_disp, score);

    // Update information
    *pos_disp_row_it = (std::is_same_v<T, float>) ? static_cast<float>(pos_row_disp) : pos_row_disp;
    *pos_disp_col_it = (std::is_same_v<T, float>) ? static_cast<float>(pos_col_disp) : pos_col_disp;
    *score_it = (std::is_same_v<T, float>) ? static_cast<float>(score) : score;
  }
}

/**
 * @brief Refine disparities by resampling the right image
 *
 * @param nb_cols : number of columns in the cost volumes
 * @param left_image : left image
 * @param right_image : right image
 * @param disparity_map_col : 1D col disparity data of size nb_row * nb_col
 * @param disparity_map_row : 1D row disparity data of size nb_row * nb_col
 * @param score_map : 1D score data of size nb_row * nb_col
 * @param invalid_map : 1D invalid (peak on edge and invalid) data of size nb_row * nb_col
 * @param subpixel : sub-sampling of cost_volume
 * @param nb_iterations : number of iterations of the dichotomy
 * @param filter : interpolation filter
 * @param method_matching_cost : matching cost measure
 * @param window_size : size of the correlation window
 * @param step : step of the matching cost
 * @param offset_cv_img : offset of the cost volume in the image
 * @param disp_min : minimum disparity values
 * @param right_valid_mask : mask of valid pixels in the right image
 */
template <typename U>
void compute_dichotomy_image(int nb_cols,
                             const P2d::Matrixf& left_image,
                             const P2d::Matrixf& right_image,
                             Eigen::Ref<U> disparity_map_col,
                             Eigen::Ref<U> disparity_map_row,
                             Eigen::Ref<U> score_map,
                             const Eigen::Ref<const P2d::VectorX<uint8_t>>& invalid_map,
                             int subpixel,
                             int nb_iterations,
                             abstractfilter::AbstractFilter& filter,
                             std::string method_matching_cost,
                             int window_size,
                             const Eigen::Vector2i& step,
                             const Eigen::Vector2i& offset_cv_img,
                             const Eigen::Vector2d& disp_min,
                             const P2d::MatrixUI& right_valid_mask) {
  // Get parameters
  const auto first_iteration = (subpixel != 4) ? subpixel : 3;  //< subpixel is 1, 2 or 4
  auto pos_disp_col_it = disparity_map_col.begin();
  auto pos_disp_row_it = disparity_map_row.begin();
  auto score_it = score_map.begin();
  auto invalid_it = invalid_map.begin();
  std::size_t index = 0;
  const P2d::MatrixD right_image_double = right_image.cast<double>();

  // Loop on each image calculated point
  for (; pos_disp_col_it != disparity_map_col.end();
       ++pos_disp_col_it, ++pos_disp_row_it, ++score_it, ++invalid_it, ++index) {
    // Taking into account the peak at the edge & invalid disparities (== 1 in the array)
    if (*invalid_it == 1)
      continue;
    // Check initial disparity is not nan
    if (std::isnan(*pos_disp_row_it) or std::isnan(*pos_disp_col_it))
      continue;

    double pos_row_disp = static_cast<double>(*pos_disp_row_it);
    double pos_col_disp = static_cast<double>(*pos_disp_col_it);
    double score = static_cast<double>(*score_it);
    // Compute pixel positions in the right image
    const std::size_t row = index / nb_cols;
    const std::size_t col = index % nb_cols;
    const int pixel_row = offset_cv_img[0] + static_cast<int>(row) * step[0];
    const int pixel_col = offset_cv_img[1] + static_cast<int>(col) * step[1];

    // Compute resampling on right image
    image_resampling(left_image, right_image_double, pixel_row, pixel_col, window_size,
                     nb_iterations, first_iteration, filter, method_matching_cost, subpixel,
                     disp_min[0], disp_min[1], pos_row_disp, pos_col_disp, score, right_valid_mask);

    // Update information
    *pos_disp_row_it = static_cast<typename U::Scalar>(pos_row_disp);
    *pos_disp_col_it = static_cast<typename U::Scalar>(pos_col_disp);
    *score_it = static_cast<typename U::Scalar>(score);
  }
}

#endif