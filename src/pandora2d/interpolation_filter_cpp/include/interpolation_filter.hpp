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
This module contains functions associated to the Abstract filter class for cpp.
*/

#ifndef INTERPOLATIONFILTER_HPP
#define INTERPOLATIONFILTER_HPP

#include <optional>

#include "constant.hpp"
#include "margins.hpp"
#include "pandora2d_type.hpp"

namespace abstractfilter {

/**
 * @brief  This abstract class allows for the instantiation of a filter.
 */

class AbstractFilter {
 public:
  /**
   * @brief Construct a new Abstract Filter object
   *
   */
  AbstractFilter() = default;

  /**
   * @brief Construct a new Abstract Filter object
   *
   * @param size size of the filter
   * @param margins margins of the fiter
   */
  AbstractFilter(int size, Margins margins);

  /**
   * @brief Destroy the Abstract Filter object
   *
   */
  ~AbstractFilter() = default;

  /**
   * @brief Get the coeffs object
   *
   * @param fractional_shift positive fractional shift of the subpixel
   * position to be interpolated
   * @return P2d::VectorD, an array of interpolator coefficients
   * whose size depends on the filter margins
   */
  virtual P2d::VectorD get_coeffs(const double fractional_shift) = 0;

  /**
   * @brief  Returns the value of the interpolated position
   *
   * @param resampling_area area on which interpolator coefficients will be applied
   * @param row_coeff interpolator coefficients in cols
   * @param col_coeff interpolator coefficients in rows
   * @return double
   */
  double apply(const P2d::MatrixD& resampling_area,
               const P2d::VectorD& row_coeff,
               const P2d::VectorD& col_coeff) const;

  /**
   * @brief
   *
   * @param image image
   * @param col_positions subpix columns positions to be interpolated
   * @param row_positions subpix rows positions to be interpolated
   * @param max_fractional_value maximum fractional value used to get coefficients
   * @return P2d::VectorD, the interpolated value of the position
   * corresponding to col_coeff and row_coeff
   */
  P2d::VectorD interpolate(const P2d::MatrixD& image,
                           const P2d::VectorD& col_positions,
                           const P2d::VectorD& row_positions,
                           const double max_fractional_value = MAX_FRACTIONAL_VALUE);

  /**
   * @brief Interpolate the union of the 8 neighbour windows around a subpixel origin
   *
   * origin is the current best point (pixel + disparity). The 8 neighbours shifted by precision
   * are interpolated on a compact grid whose sampling axis skips unused points. At precision 0.5
   * the grid size is 2 * window_size + 1; at finer precision it is 3 * window_size. Points of the
   * (0, 0) window and positions whose resampling area falls outside the image stay NaN.
   *
   * @param image image
   * @param window_size size of each correlation window
   * @param origin_row row of the current best point
   * @param origin_col col of the current best point
   * @param precision search precision (1/pow(2,n), in pixels, at the nth iteration)
   * @param right_valid_mask mask of valid pixels in the right image
   * @return std::optional<P2d::Matrixf>, the interpolated grid or std::nullopt when no position
   * can be interpolated
   */
  std::optional<P2d::Matrixf> interpolate_candidate_windows(const P2d::MatrixD& image,
                                                            int window_size,
                                                            double origin_row,
                                                            double origin_col,
                                                            double precision,
                                                            const P2d::MatrixUI& right_valid_mask);

  /**
   * @brief Get the size attribute
   *
   * @return int
   */
  int get_size() const { return m_size; }

  /**
   * @brief Get the margins attribute
   *
   * @return Margins
   */
  Margins get_margins() const { return m_margins; }

 protected:
  int m_size = 4;                 ///< filter size
  Margins m_margins{0, 0, 0, 0};  ///< filter margins

 private:
  /**
   * @brief The function builds a mapping from the compact grid to the window pixels
   *
   * @param window_size The size of the correlation window
   * @param inverse_precision The inverse of the search precision (1/precision)
   * @return Eigen::VectorXi, the mapping from the compact grid to the window pixels
   */
  Eigen::VectorXi build_grid_to_window_mapping(int window_size, int inverse_precision) const;

  /**
   * @brief Build candidate window positions for interpolation
   *
   * @param grid_index  The mapping from the compact grid to the window pixels
   * @param origin The current best point (pixel + disparity)
   * @param precision The search precision (1/pow(2,n), in pixels, at the nth iteration)
   * @param half_window Half of the correlation window size
   * @return P2d::VectorD the candidate window positions for interpolation
   */
  P2d::VectorD build_candidate_window_positions(const Eigen::VectorXi& grid_index,
                                                double origin,
                                                double precision,
                                                int half_window) const;

  /**
   * @brief Check if the filter area contains at least one invalid pixel in the right image
   *
   * @param right_valid_mask valid point mask in right image
   * @param top_left_row top left row point in filter area
   * @param top_left_col top left col point in filter area
   * @param filter_size filter size
   * @return bool true if the filter area contains only valid pixels, false otherwise
   */
  bool has_only_valid_pixels(const P2d::MatrixUI& right_valid_mask,
                             int top_left_row,
                             int top_left_col,
                             int filter_size);
};

}  // namespace abstractfilter

#endif  // namespace filter