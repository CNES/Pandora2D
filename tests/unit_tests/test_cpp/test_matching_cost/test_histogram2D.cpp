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
This module contains tests associated to histogram 2D.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "conftest.hpp"
#include "global_conftest.hpp"
#include "histogram1D.hpp"
#include "histogram2D.hpp"

TEST_CASE("Test constructor") {
  P2d::MatrixD left(4, 4);
  P2d::MatrixD right(4, 4);

  left << 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1;

  right << 1, 1, 2, 3, 1, 1, 1, 3, 1, 1, 1, 3, 3, 3, 3, 3;

  Histogram1D left_hist = Histogram1D(left);
  Histogram1D right_hist = Histogram1D(right);

  SUBCASE("With two histogram1D") {
    Histogram2D hist = Histogram2D(left_hist, right_hist);
    check_inside_eigen_element<P2d::MatrixD>(hist.values(), P2d::MatrixD::Zero(2, 2));
  }

  P2d::MatrixD values(2, 2);
  values << 8, 4, 0, 4;

  SUBCASE("With values and two histogram1D") {
    Histogram2D hist = Histogram2D(values, left_hist, right_hist);
    check_inside_eigen_element<P2d::MatrixD>(hist.values(), values);
  }
}

TEST_CASE("Test calculate_histogram2D function") {
  SUBCASE("With integer matrix") {
    P2d::MatrixD left(4, 4);
    P2d::MatrixD right(4, 4);

    left << 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1;

    right << 1, 1, 2, 3, 1, 1, 1, 3, 1, 1, 1, 3, 3, 3, 3, 3;

    P2d::MatrixD expected_values(2, 2);
    expected_values << 8, 4, 0, 4;

    Histogram2D hist = calculate_histogram2D(left, right);
    check_inside_eigen_element<P2d::MatrixD>(hist.values(), expected_values);
  }

  SUBCASE("With float matrix") {
    P2d::MatrixD left(4, 4);
    P2d::MatrixD right(4, 4);

    left << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    right << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    P2d::MatrixD expected_values(3, 3);
    expected_values << 1, 3, 1, 0, 5, 1, 2, 3, 0;

    Histogram2D hist = calculate_histogram2D(left, right);
    check_inside_eigen_element<P2d::MatrixD>(hist.values(), expected_values);
  }

  SUBCASE("With 120 bins images") {
    // Created images img_l and img_r produce histogram1D with 120 bins.
    auto img_l = create_image(std::size_t(81), 0., 0.5);
    auto img_r = create_image(std::size_t(81), 0., 0.5);
    auto hist2d = calculate_histogram2D(img_l, img_r);

    // As the number of bins of the two histograms 1D is initially greater than NB_BINS_MAX,
    // it is fixed to 100 bins for each histogram 1D.
    // Then, hist2d.values() shape is (100,100)
    CHECK(hist2d.values().size() == 100 * 100);
  }
}
