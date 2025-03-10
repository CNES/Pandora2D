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
#include "cost_volume.hpp"

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