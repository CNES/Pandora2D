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
This module contains functions associated to the CFOG (Channel Features of
Oriented Gradients) measure in C++.
*/

#ifndef CFOG_HPP
#define CFOG_HPP

#include <array>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "operation.hpp"
#include "pandora2d_type.hpp"

using RowMajorMatrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Compute a 1D Gaussian kernel of radius 3*sigma.
 *
 * @param sigma standard deviation of the Gaussian
 * @return std::vector<float> normalised kernel of size 2*(int(3*sigma))+1
 */
inline std::vector<float> gaussian_kernel_1d(float sigma) {
  int radius = int(3 * sigma);
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float sum = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
    sum += kernel[i + radius];
  }
  for (int i = 0; i < size; ++i)
    kernel[i] /= sum;
  return kernel;
}

/**
 * @brief Distribute gradient magnitude into soft orientation bins.
 *
 * For each pixel, the magnitude is split between the two nearest angular bins
 * with linear interpolation weights.  The output tensor is stored in
 * row-major order: cfog[row][col][channel] = cfog[(row*nb_rows + col)*n_channels + channel].
 *
 * @param magnitude  nb_rows×nc_cols matrix of gradient magnitudes
 * @param orientation nb_rows×nc_cols matrix of gradient orientations (radians)
 * @param n_channels  number of orientation bins
 * @param angle_range total angular range (2π for signed, π for unsigned)
 * @param cfog        output nb_rows×nc_cols×n_channels flat array (caller-allocated)
 * @param use_openmp  if false, disable parallelism (e.g. when called from an outer parallel region)
 */
inline void soft_binning(const P2d::Matrixf& magnitude,
                         const P2d::Matrixf& orientation,
                         int n_channels,
                         float angle_range,
                         std::vector<float>& cfog,
                         bool use_openmp = true) {
  const int nb_rows = magnitude.rows();
  const int nb_cols = magnitude.cols();
  const float bin_width = angle_range / n_channels;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (use_openmp)
#endif
  for (int row = 0; row < nb_rows; ++row) {
    for (int col = 0; col < nb_cols; ++col) {
      float mag = magnitude(row, col);
      float ang = orientation(row, col);

      for (int channel = 0; channel < n_channels; ++channel) {
        float center = channel * bin_width;
        float diff = std::abs(ang - center);
        if (diff > angle_range - diff)
          diff = angle_range - diff;
        float weight = 1.0f - diff / bin_width;
        cfog[(row * nb_cols + col) * n_channels + channel] = (weight > 0.0f) ? weight * mag : 0.0f;
      }
    }
  }
}

/**
 * @brief Apply a separable Gaussian blur to an nb_rows×nb_cols×nb_channels tensor (in-place).
 *
 * Borders are handled by clamping indices to valid range.
 *
 * @param cfog   flat H×nb_cols×nb_channels array, modified in place
 * @param nb_rows      number of rows
 * @param nb_cols      number of columns
 * @param nb_channels      number of channels
 * @param kernel 1D Gaussian kernel (size = 2*radius+1)
 * @param use_openmp  if false, disable parallelism (e.g. when called from an outer parallel region)
 */
inline void gaussian_blur(std::vector<float>& cfog,
                          int nb_rows,
                          int nb_cols,
                          int nb_channels,
                          const std::vector<float>& kernel,
                          bool use_openmp = true) {
  const int radius = static_cast<int>(kernel.size()) / 2;
  const int ksize = static_cast<int>(kernel.size());
  const int row_stride = nb_cols * nb_channels;

  std::vector<float> temp(nb_rows * nb_cols * nb_channels, 0.0f);

// Horizontal pass
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (use_openmp)
#endif
  for (int row = 0; row < nb_rows; ++row) {
    const float* row_in = cfog.data() + row * row_stride;
    float* row_out = temp.data() + row * row_stride;
    for (int col = 0; col < nb_cols; ++col) {
      float* out = row_out + col * nb_channels;
      for (int k = 0; k < ksize; ++k) {
        const int jj = std::clamp(col + k - radius, 0, nb_cols - 1);
        const float* in = row_in + jj * nb_channels;
        const float w = kernel[k];
        for (int c = 0; c < nb_channels; ++c)
          out[c] += in[c] * w;
      }
    }
  }

  std::fill(cfog.begin(), cfog.end(), 0.0f);

// Vertical pass
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (use_openmp)
#endif
  for (int row = 0; row < nb_rows; ++row) {
    float* row_out = cfog.data() + row * row_stride;
    for (int k = 0; k < ksize; ++k) {
      const int ii = std::clamp(row + k - radius, 0, nb_rows - 1);
      const float* row_in = temp.data() + ii * row_stride;
      const float w = kernel[k];
      for (int i = 0; i < row_stride; ++i)
        row_out[i] += row_in[i] * w;
    }
  }
}

/**
 * @brief L2-normalise each pixel descriptor vector (in-place).
 *
 * A small epsilon (1e-6) is added before the square root to avoid
 * division by zero on flat patches.
 *
 * @param cfog flat nb_rows×nb_cols×nb_channels array, modified in place
 * @param nb_rows    number of rows
 * @param nb_cols    number of columns
 * @param nb_channels    number of channels
 * @param use_openmp  if false, disable parallelism (e.g. when called from an outer parallel region)
 */
inline void l2_normalize(std::vector<float>& cfog,
                         int nb_rows,
                         int nb_cols,
                         int nb_channels,
                         bool use_openmp = true) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (use_openmp)
#endif
  for (int row = 0; row < nb_rows; ++row) {
    for (int col = 0; col < nb_cols; ++col) {
      float norm = 1e-6f;
      for (int channel = 0; channel < nb_channels; ++channel) {
        float val = cfog[(row * nb_cols + col) * nb_channels + channel];
        norm += val * val;
      }
      norm = std::sqrt(norm);
      for (int channel = 0; channel < nb_channels; ++channel)
        cfog[(row * nb_cols + col) * nb_channels + channel] /= norm;
    }
  }
}

/**
 * @brief Compute the sum of squared differences between two CFOG descriptors.
 *
 * @param a first descriptor (nb_rows×nb_cols×nb_channels flat array)
 * @param b second descriptor (same shape as a)
 * @return SSD value
 */
inline float ssd_for_cfog(const P2d::Matrixf& a, const P2d::Matrixf& b) {
  assert(a.rows() == b.rows() && a.cols() == b.cols());
  return (a.array() - b.array()).square().sum();
}

/**
 * @brief Compute the normalised cross-correlation between two CFOG descriptors.
 *
 * The result is in [0, 1], where 0 means perfect match and 2 means perfect
 * anti-correlation.  An epsilon (1e-6) is added to the denominator to avoid
 * division by zero on flat patches.
 *
 * @param a first descriptor (nb_rows×nb_cols×nb_channels flat array)
 * @param b second descriptor (same shape as a)
 * @return NCC value
 */
inline float ncc_for_cfog(const P2d::Matrixf& a, const P2d::Matrixf& b) {
  assert(a.rows() == b.rows() && a.cols() == b.cols());
  float mean_a = a.mean();
  float mean_b = b.mean();
  P2d::Matrixf ca = a.array() - mean_a;
  P2d::Matrixf cb = b.array() - mean_b;
  float denom = std::sqrt(ca.squaredNorm() * cb.squaredNorm()) + 1e-6f;
  return 1.0f - (ca.cwiseProduct(cb).sum() / denom);
}

/**
 * @brief Compute the CFOG descriptor for a single image patch.
 *
 * Gradients are computed with a simple finite-difference operator.
 * Border pixels (first/last row and column) are left at zero.
 *
 * @param patch            input nb_rows×nb_cols image patch
 * @param nb_channels       number of orientation bins (default 9)
 * @param signed_orientation use full 2π range when true, π otherwise (default true)
 * @param sigma            standard deviation for the Gaussian smoothing (default 1.0)
 * @param use_openmp       if false, disable internal parallelism — set to false when this
 *                         function is called from an outer OpenMP parallel region (e.g.
 *                         compute_cfog_cv) to avoid nested parallelism overhead.
 * @return P2d::MatrixX<float> shaped (nb_rows, nb_cols*nb_channels), where channel channel of
 *         pixel (row,col) is stored at column col*nb_channels + channel.  This layout
 *         mirrors the P2d::Matrixf convention used throughout the project.
 */
inline P2d::Matrixf descriptor_cfog(const P2d::Matrixf& patch,
                                    int nb_channels = 3,
                                    bool signed_orientation = true,
                                    float sigma = 0.5f,
                                    bool use_openmp = false) {
  // By default, use_openmp=false: descriptor_cfog is typically called from
  // the parallel loop in compute_cfog_cv. Set use_openmp=true only
  // if descriptor_cfog is called outside any parallel context.
  const int nb_rows = patch.rows();
  const int nb_cols = patch.cols();

  // Working arrays
  P2d::Matrixf gx = P2d::Matrixf::Zero(nb_rows, nb_cols);
  P2d::Matrixf gy = P2d::Matrixf::Zero(nb_rows, nb_cols);
  P2d::Matrixf magnitude = P2d::Matrixf::Zero(nb_rows, nb_cols);
  P2d::Matrixf orientation = P2d::Matrixf::Zero(nb_rows, nb_cols);
  std::vector<float> cfog(nb_rows * nb_cols * nb_channels, 0.0f);

  // --- Compute Sobel gradients ---
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (use_openmp)
#endif
  for (int row = 1; row < nb_rows - 1; ++row) {
    for (int col = 1; col < nb_cols - 1; ++col) {
      gx(row, col) = (patch(row - 1, col + 1) + 2 * patch(row, col + 1) + patch(row + 1, col + 1)) -
                     (patch(row - 1, col - 1) + 2 * patch(row, col - 1) + patch(row + 1, col - 1));
      gy(row, col) = (patch(row + 1, col - 1) + 2 * patch(row + 1, col) + patch(row + 1, col + 1)) -
                     (patch(row - 1, col - 1) + 2 * patch(row - 1, col) + patch(row - 1, col + 1));

      magnitude(row, col) = std::sqrt(gx(row, col) * gx(row, col) + gy(row, col) * gy(row, col));
      orientation(row, col) = std::atan2(gy(row, col), gx(row, col));
      if (signed_orientation)
        orientation(row, col) = std::fmod(orientation(row, col) + 2 * M_PI, 2 * M_PI);
      else
        orientation(row, col) = std::fmod(orientation(row, col) + M_PI, M_PI);
    }
  }

  const float angle_range = signed_orientation ? 2 * M_PI : M_PI;

  // --- Soft orientation binning ---
  soft_binning(magnitude, orientation, nb_channels, angle_range, cfog, use_openmp);

  // --- Gaussian smoothing ---
  const auto kernel = gaussian_kernel_1d(sigma);
  gaussian_blur(cfog, nb_rows, nb_cols, nb_channels, kernel, use_openmp);

  // --- L2 normalisation ---
  l2_normalize(cfog, nb_rows, nb_cols, nb_channels, use_openmp);

  P2d::Matrixf result =
      Eigen::Map<const RowMajorMatrixf>(cfog.data(), nb_rows, nb_cols * nb_channels);

  return result;
}

#endif