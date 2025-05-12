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
This file contains useful function definitions for matching_cost tests.
*/

#ifndef MATCHING_COST_CONFTEST_HPP
#define MATCHING_COST_CONFTEST_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "cost_volume.hpp"
#include "pandora2d_type.hpp"

namespace py = pybind11;

/**
 * @brief Create an image for tests
 *
 * @param size: image shape
 * @param mean: image mean
 * @param std: image standard deviation
 * @param nb_bins: bins number for image histogram
 */
P2d::MatrixD create_image(std::size_t size, float mean, float std, double nb_bins = 120);

/**
 * @brief Load criteria dataarray saved as a binary file containing a 1D numpy array of type uint8
 *
 * @param filename: name of binary file containing the criteria array
 * @param cv_size: cost volumes size
 */
py::array_t<uint8_t> load_criteria_dataarray(const std::string& filename,
                                             const CostVolumeSize& cv_size);

/**
 * @brief Get data_path for matching cost test data
 *
 */

extern const char* data_path_env;
extern const std::string data_path;

#endif