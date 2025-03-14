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
This module contains constant for cpp.
*/

#ifndef COMMON_CONSTANT_HPP
#define COMMON_CONSTANT_HPP

#define _USE_MATH_DEFINES  // to use math and get M_PI
#include <math.h>          /* pow */
#include <limits>          /* numeric_limits */

const double MAX_FRACTIONAL_VALUE = 1 - (1. / pow(2, 9));  ///< MAX_FRACTIONAL_VALUE=0.998046875
///< corresponds to 1-1/2**9 where 9 is the maximal number of iterations for dichotomy

const double EPSILON = std::numeric_limits<float>::epsilon();  ///< Numeric limit of float type.
///< We use the same numeric limit used by Medicis.

#endif