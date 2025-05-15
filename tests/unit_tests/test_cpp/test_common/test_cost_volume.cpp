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
This module contains tests associated to the operation functions define on operation.hpp file.
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <pybind11/embed.h>
#include "cost_volume.hpp"
#include "global_conftest.hpp"

TEST_CASE("Cost volume size") {
  SUBCASE("First constructor") {
    CostVolumeSize cv_size = CostVolumeSize(2, 3, 1, 1);
    CHECK(cv_size.size() == 6);
    CHECK(cv_size.nb_disps() == 1);
  }

  SUBCASE("Second constructor") {
    CostVolumeSize cv_size = CostVolumeSize();
    CHECK(cv_size.size() == 0);
    CHECK(cv_size.nb_disps() == 0);
  }

  SUBCASE("Third constructor") {
    P2d::VectorD vec_size{{4, 5, 2, 1}};
    CostVolumeSize cv_size = CostVolumeSize(vec_size);
    CHECK(cv_size.size() == 40);
    CHECK(cv_size.nb_disps() == 2);
  }

  SUBCASE("Fourth constructor") {
    std::vector<size_t> vec_size{4, 5, 2, 1};
    CostVolumeSize cv_size = CostVolumeSize(vec_size);
    CHECK(cv_size.size() == 40);
    CHECK(cv_size.nb_disps() == 2);
  }
}

/**
 * Create cost volume with Eigen
 */
P2d::Matrixf cost_volume_4_4_2_3() {
  // Cost volume for (4,4) image and row disp [4, 5] and col disp [-2, -1, 0]
  // pixel-by-pixel filling
  unsigned int cv_size = 4 * 4 * 2 * 3;
  P2d::Vectorf cost_volume(cv_size);
  // clang-format off
  cost_volume << 0.680375, 0.211234, 0.566198, 0.59688, 0.823295, 0.604897,
                 0.329554, 0.536459, 0.444451, 0.10794, 0.045205, 0.257742,
                 0.270431, 0.026801, 0.904459, 0.83239, 0.271423, 0.434594,
                 0.716795, 0.213938, 0.967399, 0.51422, 0.725537, 0.608354,
                 0.686642, 0.198111, 0.740419, 0.78238, 0.997849, 0.563486,
                 0.025864, 0.678224, 0.22528,  0.40793, 0.275105, 0.0485744,
                 0.012834, 0.94555,  0.414966, 0.54271, 0.05349,  0.539828,
                 0.199543, 0.783059, 0.433371, 0.29508, 0.615449, 0.838053,
                 0.860489, 0.898654, 0.051990, 0.82788, 0.615572, 0.326454,
                 0.780465, 0.302214, 0.871657, 0.95995, 0.084596, 0.873808,
                 0.52344, 0.941268,  0.804416, 0.70184, 0.466669, 0.0795207,
                 0.249586, 0.520497, 0.025070, 0.33544, 0.063212, 0.921439,
                 0.124725, 0.86367,  0.86162,  0.44190, 0.431413, 0.477069,
                 0.279958, 0.291903, 0.375723, 0.66805, 0.119791, 0.76015,
                 0.658402, 0.339326, 0.542064, 0.78674, 0.29928,  0.37334,
                 0.912937, 0.17728, 0.314608, 0.71735, 0.12088, 0.84794;
  // clang-format on
  return cost_volume;
}

/**
 * Create cost volume with pybind array
 *
 */
py::array_t<float> py_cost_volume_4_4_2_3() {
  // Cost volume for (4,4) image and row disp [4, 5] and col disp [-2, -1, 0]
  py::scoped_interpreter guard{};
  // define py_array
  std::vector<size_t> shape = {4, 4, 2, 3};
  py::array_t<float> cost_volume = py::array_t<float>(shape);
  CostVolumeSize cv_size = CostVolumeSize(shape);

  // data access
  auto unchecked_cv = cost_volume.mutable_unchecked<4>();

  // init std vector with value
  // pixel-by-pixel filling
  // clang-format off
  std::vector<std::vector<std::vector<std::vector<float>>>> data = {
    {
      {{0.680375, 0.211234, 0.566198}, {0.59688, 0.823295, 0.604897}},  // pixel (0, 0)
      {{0.329554, 0.536459, 0.444451}, {0.10794, 0.045205, 0.257742}},
      {{0.270431, 0.026801, 0.904459}, {0.83239, 0.271423, 0.434594}},
      {{0.716795, 0.213938, 0.967399}, {0.51422, 0.725537, 0.608354}}
    },
    {
      {{0.686642, 0.198111, 0.740419}, {0.78238, 0.997849, 0.563486}},
      {{0.025864, 0.678224, 0.22528}, {0.40793, 0.275105, 0.0485744}},
      {{0.012834, 0.94555,  0.414966}, {0.54271, 0.05349,  0.539828}},
      {{0.199543, 0.783059, 0.433371}, {0.29508, 0.615449, 0.838053}}
    },
    {
      {{0.860489, 0.898654, 0.051990}, {0.82788, 0.615572, 0.326454}},
      {{0.780465, 0.302214, 0.871657}, {0.95995, 0.084596, 0.873808}},
      {{0.52344, 0.941268,  0.804416}, {0.70184, 0.466669, 0.0795207}},
      {{0.249586, 0.520497, 0.025070}, {0.33544, 0.063212, 0.921439}}
    },
    {
      {{0.124725, 0.86367,  0.86162}, {0.44190, 0.431413, 0.477069}},
      {{0.279958, 0.291903, 0.375723}, {0.66805, 0.119791, 0.76015}},
      {{0.658402, 0.339326, 0.542064}, {0.78674, 0.29928,  0.37334}},
      {{0.912937, 0.17728, 0.314608}, {0.71735, 0.12088, 0.84794}}
    }
  };
  // clang-format on

  // fill py_array element
  for (size_t i = 0; i < cv_size.nb_row; ++i) {
    for (size_t j = 0; j < cv_size.nb_col; ++j) {
      for (size_t k = 0; k < cv_size.nb_disp_row; ++k) {
        for (size_t l = 0; l < cv_size.nb_disp_col; ++l) {
          unchecked_cv(i, j, k, l) = data[i][j][k][l];
        }
      }
    }
  }
  return cost_volume;
}

TEST_CASE("Position2D") {
  SUBCASE("First constructor") {
    unsigned int row = 2;
    unsigned int column = 3;
    Position2D p = Position2D(2, 3);
    CHECK(p.row == row);
    CHECK(p.col == column);
  }

  SUBCASE("Second constructor") {
    Position2D p = Position2D();
    CHECK(p.row == 0u);
    CHECK(p.col == 0u);
  }
}

TEST_CASE("get_cost_surfaces") {
  CostVolumeSize cv_size = CostVolumeSize(4, 4, 2, 3);
  auto cost_volume = py_cost_volume_4_4_2_3();

  // Check First pixel
  Position2D pixel = Position2D();
  P2d::Matrixf expected(cv_size.nb_disp_row, cv_size.nb_disp_col);
  expected << 0.680375, 0.211234, 0.566198, 0.59688, 0.823295, 0.604897;
  check_inside_eigen_element<P2d::Matrixf>(
      get_cost_surface<float, float>(cost_volume, position2d_to_index(pixel, cv_size), cv_size),
      expected);

  // Check pixel at (1, 2)
  pixel = Position2D(1, 2);
  expected << 0.012834, 0.94555, 0.414966, 0.54271, 0.05349, 0.539828;
  check_inside_eigen_element<P2d::Matrixf>(
      get_cost_surface<float, float>(cost_volume, position2d_to_index(pixel, cv_size), cv_size),
      expected);

  // Check pixel at (2, 3)
  pixel = Position2D(2, 3);
  expected << 0.249586, 0.520497, 0.025070, 0.33544, 0.063212, 0.921439;
  check_inside_eigen_element<P2d::Matrixf>(
      get_cost_surface<float, float>(cost_volume, position2d_to_index(pixel, cv_size), cv_size),
      expected);

  // Check Last pixel
  pixel = Position2D(3, 3);
  expected << 0.912937, 0.17728, 0.314608, 0.71735, 0.12088, 0.84794;
  check_inside_eigen_element<P2d::Matrixf>(
      get_cost_surface<float, float>(cost_volume, position2d_to_index(pixel, cv_size), cv_size),
      expected);
}
