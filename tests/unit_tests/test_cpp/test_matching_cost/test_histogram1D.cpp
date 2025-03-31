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
This module contains tests associated to histogram 1D.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "bin.hpp"
#include "conftest.hpp"
#include "global_conftest.hpp"
#include "histogram1D.hpp"

TEST_CASE("Test constructor") {
  SUBCASE("With P2d::VectorD::Zero") {
    P2d::VectorD m = P2d::VectorD::Zero(3);
    Histogram1D hist = Histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), P2d::VectorD::Zero(1));
    CHECK(hist.nb_bins() == 1);
    CHECK(hist.low_bound() == -0.5);
    CHECK(hist.bins_width() == 1);
  }

  SUBCASE("With P2d::VectorD {1,2,3,4}") {
    P2d::VectorD m(4);
    m << 1, 2, 3, 4;

    Histogram1D hist = Histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), P2d::VectorD::Zero(2));
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
  }

  SUBCASE("First constructor") {
    P2d::VectorD m = P2d::VectorD::Ones(3);
    Histogram1D hist = Histogram1D(m, 3, 0.1, 1.3);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), m);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == 0.1);
    CHECK(hist.bins_width() == 1.3);
  }

  SUBCASE("Second constructor") {
    Histogram1D hist = Histogram1D(2, 0.1, 1.3);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), P2d::VectorD::Zero(2));
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == 0.1);
    CHECK(hist.bins_width() == 1.3);
  }
}

TEST_CASE("Test calculate_histogram1D function") {
  SUBCASE("positive low_bound & matrix coefficients") {
    P2d::MatrixD m(1, 4);
    m << 1, 2, 3, 4;

    auto hist = calculate_histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), P2d::VectorD::Ones(2) * 2);
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
  }

  SUBCASE("negative low_bound & positive matrix coefficients") {
    P2d::MatrixD m(4, 4);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD hist_expected(3, 1);
    hist_expected << 5, 6, 5;

    auto hist = calculate_histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), hist_expected);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == doctest::Approx(-1.0795972).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(6.3863981).epsilon(1e-7));
  }

  SUBCASE("negative low_bound & matrix coefficients") {
    P2d::MatrixD m(1, 4);
    m << -11, -12, -13, -14;

    auto hist = calculate_histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), P2d::VectorD::Ones(2) * 2);
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(-14.9587716).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
  }

  SUBCASE("positive & negative matrix coefficients") {
    P2d::MatrixD m(4, 4);
    m << -0.1, -0.2, 0.30, 0.40, 0.1, 0.3, -0.45, -0.59, 0.99, -0.101, 0.11452, 0.1235, -0.36,
        -0.256, -0.56, -0.1598;

    P2d::MatrixD hist_expected(3, 1);
    hist_expected << 9, 6, 1;

    auto hist = calculate_histogram1D(m);

    check_inside_eigen_element<P2d::VectorD>(hist.values(), hist_expected);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == doctest::Approx(-0.6199559).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(0.5466373).epsilon(1e-7));
  }

  SUBCASE("test with a 120 bins image") {
    auto m = create_image(std::size_t(81), 0., 0.5);
    auto hist = calculate_histogram1D(m);

    auto bins_width = get_bins_width(m);
    auto dynamic_range = m.maxCoeff() - m.minCoeff();
    auto nb_bins = static_cast<int>(1. + (dynamic_range / bins_width));
    auto low_bound =
        m.minCoeff() - (static_cast<double>(nb_bins) * bins_width - dynamic_range) / 2.;

    CHECK(hist.nb_bins() == 100);
    CHECK(hist.low_bound() <= low_bound);
    CHECK(hist.bins_width() >= bins_width);
  }
}
