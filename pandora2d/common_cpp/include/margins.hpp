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
This module contains Margins struct and associated operators.
*/

#ifndef COMMON_MARGINS_HPP
#define COMMON_MARGINS_HPP

#include <iostream>

/**
 * @brief Margins struct
 *
 *
 */
struct Margins {
  /**
   * @brief Construct a new Margins object
   *
   * @param left_
   * @param up_
   * @param right_
   * @param down_
   */
  Margins(int left_, int up_, int right_, int down_)
      : left(left_), up(up_), right(right_), down(down_) {}

  int left;   ///< left margins attribute
  int up;     ///< up margins attribute
  int right;  ///< right margins attribute
  int down;   ///< down margins attribute

  /**
   * @brief Operator == for Margins
   *
   * @param m1 Margins
   * @param m2 Margins
   * @return true or false
   */
  friend bool operator==(Margins const& m1, Margins const& m2) {
    return ((m1.left == m2.left) & (m1.up == m2.up) & (m1.right == m2.right) &
            (m1.down == m2.down));
  }

  /**
   * @brief Operator << for margins
   *
   * @param output std::ostream &
   * @param m const Margins &
   * @return std::ostream&
   */
  friend std::ostream& operator<<(std::ostream& output, const Margins& m) {
    output << "{ " << m.left << " " << m.up << " " << m.right << " " << m.down << " }";

    output << std::endl;
    return output;
  }
};

#endif