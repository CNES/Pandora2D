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
This module contains tests associated to mutual information computation.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "mutual_information.hpp"

/**
 * @brief Entropy calculation medicis version
 * The entropy1D is calculated here:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * histo1d_d.c#L400
 * The entropy2D is calculated here:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * histo2d_d.c#L297
 * @tparam T hist1D or hist2D
 * @param nb_pixel of the image
 * @param hist to iterate
 */
template <typename T, typename TypeHist>
T get_entropy_medicis(const T nb_pixel, const TypeHist& hist) {
  T entropy = std::log2(nb_pixel);

  for (auto bin_value : hist.values().reshaped()) {
    if (bin_value != 0.) {
      entropy -= bin_value / nb_pixel * std::log2(bin_value);
    }
  };

  // Entropy cannot be negative
  return entropy < 0. ? 0. : entropy;
};

/**
 * @brief Compute entropy 1D medicis version
 * @param img
 * @return T entropy1D
 */
template <typename T>
T calculate_entropy1D_medicis(const P2d::MatrixX<T>& img) {
  auto nb_pixel = static_cast<T>(img.size());
  auto hist_1D = calculate_histogram1D(img);

  return get_entropy_medicis<T, Histogram1D<T>>(nb_pixel, hist_1D);
};

/**
 * @brief Compute entropy 2D medicis version
 * @param img_l left image
 * @param img_r right image
 * @return T entropy 2D
 */
template <typename T>
T calculate_entropy2D_medicis(const P2d::MatrixX<T>& img_l, const P2d::MatrixX<T>& img_r) {
  // same size for left and right images
  auto nb_pixel = static_cast<T>(img_l.size());
  auto hist_2D = calculate_histogram2D(img_l, img_r);

  return get_entropy_medicis<T, Histogram2D<T>>(nb_pixel, hist_2D);
};

template <typename type, typename vector_type, typename matrix_type>
struct TypeStruct {
  using Type = type;
  using VectorType = vector_type;
  using MatrixType = matrix_type;
};

TYPE_TO_STRING_AS("Float", TypeStruct<float, P2d::Vectorf, P2d::Matrixf>);
TYPE_TO_STRING_AS("Double", TypeStruct<double, P2d::VectorD, P2d::MatrixD>);

TEST_CASE_TEMPLATE("Test Entropy1D",
                   T,
                   TypeStruct<float, P2d::Vectorf, P2d::Matrixf>,
                   TypeStruct<double, P2d::VectorD, P2d::MatrixD>) {
  using Type = typename T::Type;
  using VectorType = typename T::VectorType;
  using MatrixType = typename T::MatrixType;

  SUBCASE("4x4 matrix") {
    MatrixType img(4, 4);
    img << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    Type entropy_gt = 1.579434;
    Type entropy_medicis = calculate_entropy1D_medicis<Type>(img);
    Type entropy_1D_img = calculate_entropy1D<Type>(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Second 4x4 matrix") {
    MatrixType img(4, 4);
    img << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    Type entropy_gt = 1.19946029;
    Type entropy_medicis = calculate_entropy1D_medicis<Type>(img);
    Type entropy_1D_img = calculate_entropy1D<Type>(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with negative values") {
    MatrixType img(4, 4);
    img << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    Type entropy_gt = 1.5052408;
    Type entropy_medicis = calculate_entropy1D_medicis<Type>(img);
    Type entropy_1D_img = calculate_entropy1D<Type>(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with all identical values") {
    MatrixType img(4, 4);
    img << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    Type entropy_gt = 0.;
    Type entropy_medicis = calculate_entropy1D_medicis<Type>(img);
    Type entropy_1D_img = calculate_entropy1D<Type>(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Vector of size 5") {
    VectorType img(5);
    img << 1, 5, 12, 4, 0;

    Type entropy_gt = 0.7219280;
    Type entropy_medicis = calculate_entropy1D_medicis<Type>(img);
    Type entropy_1D_img = calculate_entropy1D<Type>(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }
}

TEST_CASE_TEMPLATE("Test Entropy2D",
                   T,
                   TypeStruct<float, P2d::Vectorf, P2d::Matrixf>,
                   TypeStruct<double, P2d::VectorD, P2d::MatrixD>) {
  using Type = typename T::Type;
  using VectorType = typename T::VectorType;
  using MatrixType = typename T::MatrixType;

  SUBCASE("4x4 matrix") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    Type entropy_gt = 2.55503653;
    Type entropy_medicis = calculate_entropy2D_medicis<Type>(img_l, img_r);
    Type entropy_2D_img = calculate_entropy2D<Type>(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with negative values in right img") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    Type entropy_gt = 3.0306390;
    Type entropy_medicis = calculate_entropy2D_medicis<Type>(img_l, img_r);
    Type entropy_2D_img = calculate_entropy2D<Type>(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with identical values in right img") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    Type entropy_gt = 1.579434;
    Type entropy_medicis = calculate_entropy2D_medicis<Type>(img_l, img_r);
    Type entropy_2D_img = calculate_entropy2D<Type>(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Vectors of size 5") {
    VectorType img_l(5);
    img_l << 1, 5, 12, 4, 0;

    VectorType img_r(5);
    img_r << 2, 4, 18, 9, 25;

    Type entropy_gt = 1.3709505;
    Type entropy_medicis = calculate_entropy2D_medicis<Type>(img_l, img_r);
    Type entropy_2D_img = calculate_entropy2D<Type>(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }
}

TEST_CASE_TEMPLATE("Test MutualInformation",
                   T,
                   TypeStruct<float, P2d::Vectorf, P2d::Matrixf>,
                   TypeStruct<double, P2d::VectorD, P2d::MatrixD>) {
  using Type = typename T::Type;
  using VectorType = typename T::VectorType;
  using MatrixType = typename T::MatrixType;

  SUBCASE("4x4 matrix") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    Type mutual_information_gt = 1.579434 + 1.19946029 - 2.55503653;  // 0.22385776
    Type mutual_information = calculate_mutual_information<Type>(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Identical images") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0;

    MatrixType img_r(4, 4);
    img_r << 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0;

    Type mutual_information_gt = 1.0;
    Type mutual_information = calculate_mutual_information<Type>(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Negative values in right image") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    Type mutual_information_gt = 1.579434 + 1.5052408 - 3.0306390;  // 0.05403575564
    Type mutual_information = calculate_mutual_information<Type>(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-6));
  }

  SUBCASE("Identical values in right image") {
    MatrixType img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    MatrixType img_r(4, 4);
    img_r << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    Type mutual_information_gt = 0.;
    Type mutual_information = calculate_mutual_information<Type>(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Vector of size 5") {
    VectorType img_l(5);
    img_l << 1, 5, 12, 4, 0;

    VectorType img_r(5);
    img_r << 2, 4, 18, 9, 25;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    Type mutual_information_gt = 0.7219280 + 0.97095059 - 1.3709505;  // 0.32192809489
    Type mutual_information = calculate_mutual_information<Type>(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }
}