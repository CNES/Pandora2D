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

#include "conftest.hpp"
#include <fstream>

/**
 * Load criteria dataarray saved as an 1D numpy array of type uint8
 */
py::array_t<uint8_t> load_criteria_dataarray(const std::string& filename,
                                             const CostVolumeSize& cv_size) {
  std::ifstream file(filename, std::ios::binary);
  // Get size of file
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read numpy array data in
  std::vector<uint8_t> data(fileSize);
  file.read(reinterpret_cast<char*>(data.data()), fileSize);

  // Convert in py::array_t<uint8_t>
  const std::vector<size_t> cv_shape = {cv_size.nb_row, cv_size.nb_col, cv_size.nb_disp_row,
                                        cv_size.nb_disp_col};
  py::array_t<uint8_t> criteria_values(cv_shape);
  std::memcpy(criteria_values.mutable_data(), data.data(), data.size());

  return criteria_values;
}

/**
 * @brief Get data_path for matching cost test data
 *
 */
const char* data_path_env = std::getenv("DATA_PATH");
const std::string data_path = std::string(data_path_env ? data_path_env : "");