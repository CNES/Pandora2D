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
This module contains tests associated to the Cardinal Sine filter class for cpp.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "cardinal_sine.hpp"

TEST_SUITE("find_or_throw") {
  TEST_CASE("find") {
    P2d::VectorD container{{0., 0.25, 0.5, 0.75}};
    Eigen::Index result = find_or_throw(0.25, container, "Not found.");
    CHECK(result == 1);
  }

  TEST_CASE("throw") {
    P2d::VectorD container{{0., 0.25, 0.5, 0.75}};
    std::string message("My message.");
    CHECK_THROWS_WITH_AS(find_or_throw(1, container, message), message.data(),
                         std::invalid_argument);
  }
}

TEST_SUITE("Sinc") {
  TEST_CASE("0.1") {
    double result = sinc(0.1);
    CHECK(result == 0.983631643083466);
  }

  TEST_CASE("0") {
    double result = sinc(0.1);
    CHECK(result == doctest::Approx(0.983631643083466));
  }
}

TEST_SUITE("fractional range") {
  TEST_CASE("0.25") {
    P2d::VectorD expected{{0., 0.25, 0.5, 0.75}};
    P2d::VectorD result = fractional_range(0.25);
    CHECK(result.isApprox(expected));
  }

  TEST_CASE("0.3") {
    P2d::VectorD expected{{0., 0.3, 0.6, 0.9}};
    P2d::VectorD result = fractional_range(0.3);
    CHECK(result.isApprox(expected));
  }
}

TEST_SUITE("compute_coefficient_table") {
  TEST_CASE("Check values") {
    Eigen::Vector4d fractional_shifts{0., 0.25, 0.5, 0.75};

    P2d::MatrixD expected{
        {-7.27961266e-20, 4.96470926e-19, -2.38826216e-18, 8.10350309e-18, -1.93939483e-17,
         3.27387650e-17, 1.00000000e+00, 3.27387650e-17, -1.93939483e-17, 8.10350309e-18,
         -2.38826216e-18, 4.96470926e-19, -7.27961266e-20},
        {3.94069129e-05, -3.49122646e-04, 2.26378883e-03, -1.09605319e-02, 4.13452143e-02,
         -1.37086016e-01, 8.90555907e-01, 2.72044901e-01, -7.53645464e-02, 2.18664538e-02,
         -5.15689539e-03, 9.23500523e-04, -1.22060938e-04},
        {3.07223055e-05, -2.94841564e-04, 2.06407629e-03, -1.07216026e-02, 4.27739994e-02,
         -1.43292465e-01, 6.09455472e-01, 6.09455472e-01, -1.43292465e-01, 4.27739994e-02,
         -1.07216026e-02, 2.06407629e-03, -2.94841564e-04},
        {1.17346250e-05, -1.22064316e-04, 9.23526079e-04, -5.15703809e-03, 2.18670589e-02,
         -7.53666320e-02, 2.72052430e-01, 8.90580552e-01, -1.37089810e-01, 4.13463585e-02,
         -1.09608352e-02, 2.26385148e-03, -3.49132307e-04}};

    auto result = compute_coefficient_table(6, fractional_shifts);

    CHECK(result.isApprox(expected, 1e-6));
  }

  TEST_CASE("Check other") {
    Eigen::Vector2d fractional_shifts{0.5, 0.75};

    auto result = compute_coefficient_table(4, fractional_shifts);

    CHECK(result.rows() == 2);
    CHECK(result.cols() == 9);
  }
}

TEST_SUITE("CardinalSine") {
  TEST_CASE("Default constructor") {
    P2d::VectorD expected{{1.17346250e-05, -1.22064316e-04, 9.23526079e-04, -5.15703809e-03,
                           2.18670589e-02, -7.53666320e-02, 2.72052430e-01, 8.90580552e-01,
                           -1.37089810e-01, 4.13463585e-02, -1.09608352e-02, 2.26385148e-03,
                           -3.49132307e-04}};

    auto filter = CardinalSine();
    auto result = filter.get_coeffs(0.75);

    CHECK(result.cols() == 1);
    CHECK(result.rows() == (1 + 2 * 6));
    CHECK(result.isApprox(expected, 1e-6));
    CHECK(filter.get_size() == 13);
  }

  TEST_CASE("Parmetrized constructor") {
    auto filter = CardinalSine(7, 0.5);
    auto result = filter.get_coeffs(0.5);
    CHECK(result.cols() == 1);
    CHECK(result.rows() == (1 + 2 * 7));
  }

  TEST_CASE("Out of bound fractional_shift") {
    auto filter = CardinalSine(6, 0.25);

    CHECK_THROWS_WITH_AS(filter.get_coeffs(0.9), "Unknown fractional shift: 0.900000",
                         std::invalid_argument);
  }
}
