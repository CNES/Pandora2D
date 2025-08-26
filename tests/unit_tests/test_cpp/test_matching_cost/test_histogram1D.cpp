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

template <typename type, typename vector_type>
struct VectorTypePair {
  using Type = type;
  using VectorType = vector_type;
};

TYPE_TO_STRING_AS("Float", VectorTypePair<float, P2d::Vectorf>);
TYPE_TO_STRING_AS("Double", VectorTypePair<double, P2d::VectorD>);

TEST_CASE_TEMPLATE("Test constructor",
                   T,
                   VectorTypePair<float, P2d::Vectorf>,
                   VectorTypePair<double, P2d::VectorD>) {
  using Type = typename T::Type;
  using VectorType = typename T::VectorType;

  SUBCASE("With VectorType::Zero") {
    VectorType m = VectorType::Zero(3);
    Histogram1D hist = Histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), VectorType::Zero(1));
    CHECK(hist.nb_bins() == 1);
    CHECK(hist.low_bound() == -0.5);
    CHECK(hist.bins_width() == 1);
  }

  SUBCASE("With VectorType {1,2,3,4}") {
    VectorType m(4);
    m << 1, 2, 3, 4;

    Histogram1D hist = Histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), VectorType::Zero(2));
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-6));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
  }

  SUBCASE("First constructor") {
    VectorType m = VectorType::Ones(3);
    Histogram1D hist = Histogram1D<Type>(m, 3, 0.1, 1.3);

    check_inside_eigen_element<VectorType>(hist.values(), m);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == doctest::Approx(0.1).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(1.3).epsilon(1e-7));
  }

  SUBCASE("Second constructor") {
    Histogram1D hist = Histogram1D<Type>(2, 0.1, 1.3);

    check_inside_eigen_element<VectorType>(hist.values(), VectorType::Zero(2));
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(0.1).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(1.3).epsilon(1e-7));
  }
}

template <typename type, typename vector_type, typename matrix_type>
struct TypeStruct {
  using Type = type;
  using VectorType = vector_type;
  using MatrixType = matrix_type;
};

TYPE_TO_STRING_AS("Float", TypeStruct<float, P2d::Vectorf, P2d::Matrixf>);
TYPE_TO_STRING_AS("Double", TypeStruct<double, P2d::VectorD, P2d::MatrixD>);

TEST_CASE_TEMPLATE("Test calculate_histogram1D function",
                   T,
                   TypeStruct<float, P2d::Vectorf, P2d::Matrixf>,
                   TypeStruct<double, P2d::VectorD, P2d::MatrixD>) {
  using Type = typename T::Type;
  using VectorType = typename T::VectorType;
  using MatrixType = typename T::MatrixType;

  SUBCASE("positive low_bound & matrix coefficients") {
    MatrixType m(1, 4);
    m << 1, 2, 3, 4;

    auto hist = calculate_histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), VectorType::Ones(2) * 2);
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-6));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
    CHECK(hist.low_bound() < m.minCoeff());
    CHECK(hist.up_bound() > m.maxCoeff());
  }

  SUBCASE("negative low_bound & positive matrix coefficients") {
    MatrixType m(4, 4);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType hist_expected(3, 1);
    hist_expected << 5, 6, 5;

    auto hist = calculate_histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), hist_expected);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == doctest::Approx(-1.0795972).epsilon(1e-6));
    CHECK(hist.bins_width() == doctest::Approx(6.3863981).epsilon(1e-7));
    CHECK(hist.low_bound() < m.minCoeff());
    CHECK(hist.up_bound() > m.maxCoeff());
  }

  SUBCASE("negative low_bound & matrix coefficients") {
    MatrixType m(1, 4);
    m << -11, -12, -13, -14;

    auto hist = calculate_histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), VectorType::Ones(2) * 2);
    CHECK(hist.nb_bins() == 2);
    CHECK(hist.low_bound() == doctest::Approx(-14.9587716).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
    CHECK(hist.low_bound() < m.minCoeff());
    CHECK(hist.up_bound() > m.maxCoeff());
  }

  SUBCASE("positive & negative matrix coefficients") {
    MatrixType m(4, 4);
    m << -0.1, -0.2, 0.30, 0.40, 0.1, 0.3, -0.45, -0.59, 0.99, -0.101, 0.11452, 0.1235, -0.36,
        -0.256, -0.56, -0.1598;

    MatrixType hist_expected(3, 1);
    hist_expected << 9, 6, 1;

    auto hist = calculate_histogram1D<Type>(m);

    check_inside_eigen_element<VectorType>(hist.values(), hist_expected);
    CHECK(hist.nb_bins() == 3);
    CHECK(hist.low_bound() == doctest::Approx(-0.6199559).epsilon(1e-7));
    CHECK(hist.bins_width() == doctest::Approx(0.5466373).epsilon(1e-7));
    CHECK(hist.low_bound() < m.minCoeff());
    CHECK(hist.up_bound() > m.maxCoeff());
  }

  SUBCASE("test with a 120 bins image (nb_bins > NB_BINS_MAX)") {
    auto m = create_image<Type>(std::size_t(81), 0., 0.5);
    auto hist = calculate_histogram1D<Type>(m);

    CHECK(hist.nb_bins() == 100);
    // When nb_bins > NB_BINS_MAX,
    // the histogram lower bound is greater than the image minimum coefficient,
    // and the histogram upper bound is smaller than the image maximum coefficient.
    CHECK(hist.low_bound() > m.minCoeff());
    CHECK(hist.up_bound() < m.maxCoeff());
  }
}
