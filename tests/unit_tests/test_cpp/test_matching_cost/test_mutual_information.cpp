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
template <typename T>
double get_entropy_medicis(const double nb_pixel, const T& hist) {
  double entropy = std::log2(nb_pixel);

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
 * @return double entropy1D
 */
double calculate_entropy1D_medicis(const P2d::MatrixD& img) {
  auto nb_pixel = static_cast<double>(img.size());
  auto hist_1D = calculate_histogram1D(img);

  return get_entropy_medicis<Histogram1D>(nb_pixel, hist_1D);
};

/**
 * @brief Compute entropy 2D medicis version
 * @param img_l left image
 * @param img_r right image
 * @return double entropy 2D
 */
double calculate_entropy2D_medicis(const P2d::MatrixD& img_l, const P2d::MatrixD& img_r) {
  // same size for left and right images
  auto nb_pixel = static_cast<double>(img_l.size());
  auto hist_2D = calculate_histogram2D(img_l, img_r);

  return get_entropy_medicis<Histogram2D>(nb_pixel, hist_2D);
};

TEST_CASE("Test Entropy1D") {
  SUBCASE("4x4 matrix") {
    P2d::MatrixD img(4, 4);
    img << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    double entropy_gt = 1.579434;
    double entropy_medicis = calculate_entropy1D_medicis(img);
    double entropy_1D_img = calculate_entropy1D(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Second 4x4 matrix") {
    P2d::MatrixD img(4, 4);
    img << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    double entropy_gt = 1.19946029;
    double entropy_medicis = calculate_entropy1D_medicis(img);
    double entropy_1D_img = calculate_entropy1D(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with negative values") {
    P2d::MatrixD img(4, 4);
    img << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    double entropy_gt = 1.5052408;
    double entropy_medicis = calculate_entropy1D_medicis(img);
    double entropy_1D_img = calculate_entropy1D(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with all identical values") {
    P2d::MatrixD img(4, 4);
    img << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    double entropy_gt = 0.;
    double entropy_medicis = calculate_entropy1D_medicis(img);
    double entropy_1D_img = calculate_entropy1D(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Vector of size 5") {
    P2d::VectorD img(5);
    img << 1, 5, 12, 4, 0;

    double entropy_gt = 0.7219280;
    double entropy_medicis = calculate_entropy1D_medicis(img);
    double entropy_1D_img = calculate_entropy1D(img);

    CHECK(entropy_1D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_1D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }
}

TEST_CASE("Test Entropy2D") {
  SUBCASE("4x4 matrix") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    double entropy_gt = 2.55503653;
    double entropy_medicis = calculate_entropy2D_medicis(img_l, img_r);
    double entropy_2D_img = calculate_entropy2D(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with negative values in right img") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    double entropy_gt = 3.0306390;
    double entropy_medicis = calculate_entropy2D_medicis(img_l, img_r);
    double entropy_2D_img = calculate_entropy2D(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("4x4 matrix with identical values in right img") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    double entropy_gt = 1.579434;
    double entropy_medicis = calculate_entropy2D_medicis(img_l, img_r);
    double entropy_2D_img = calculate_entropy2D(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }

  SUBCASE("Vectors of size 5") {
    P2d::VectorD img_l(5);
    img_l << 1, 5, 12, 4, 0;

    P2d::VectorD img_r(5);
    img_r << 2, 4, 18, 9, 25;

    double entropy_gt = 1.3709505;
    double entropy_medicis = calculate_entropy2D_medicis(img_l, img_r);
    double entropy_2D_img = calculate_entropy2D(img_l, img_r);

    CHECK(entropy_2D_img == doctest::Approx(entropy_medicis).epsilon(1e-7));
    CHECK(entropy_2D_img == doctest::Approx(entropy_gt).epsilon(1e-7));
  }
}

TEST_CASE("Test MutualInformation") {
  SUBCASE("4x4 matrix") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << 1., 2., 3., 4., 2., 2., 2., 2., 4., 3., 2., 1., 1., 3., 3., 3.;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    double mutual_information_gt = 1.579434 + 1.19946029 - 2.55503653;  // 0.22385776
    double mutual_information = calculate_mutual_information(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Identical images") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0;

    P2d::MatrixD img_r(4, 4);
    img_r << 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0;

    double mutual_information_gt = 1.0;
    double mutual_information = calculate_mutual_information(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Negative values in right image") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << -2.0, -3.0, 10.0, -9.0, -11.0, -1.0, -2.0, -12.0, -5.0, 3.0, -13.0, -6.0, 6.0, -11.0,
        -4.0, -8.0;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    double mutual_information_gt = 1.579434 + 1.5052408 - 3.0306390;  // 0.05403575564
    double mutual_information = calculate_mutual_information(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Identical values in right image") {
    P2d::MatrixD img_l(4, 4);
    img_l << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

    P2d::MatrixD img_r(4, 4);
    img_r << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    double mutual_information_gt = 0.;
    double mutual_information = calculate_mutual_information(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }

  SUBCASE("Vector of size 5") {
    P2d::VectorD img_l(5);
    img_l << 1, 5, 12, 4, 0;

    P2d::VectorD img_r(5);
    img_r << 2, 4, 18, 9, 25;

    // mutual_information = E1D(img_l) + E1D(img_r) - E2D(img_l, img_r)
    double mutual_information_gt = 0.7219280 + 0.97095059 - 1.3709505;  // 0.32192809489
    double mutual_information = calculate_mutual_information(img_l, img_r);
    CHECK(mutual_information == doctest::Approx(mutual_information_gt).epsilon(1e-7));
  }
}