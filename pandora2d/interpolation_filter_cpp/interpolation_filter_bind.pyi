# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA2D
#
#     https://github.com/CNES/Pandora2D
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: skip-file

from typing import List
import numpy as np

class Margins:
    up: int
    down: int
    left: int
    right: int

class AbstractFilter:
    def __init__(self) -> None: ...
    def get_coeffs(self, fractional_shift: float) -> np.ndarray:
        """
        Returns the interpolator coefficients to be applied to the resampling area.

        The size of the returned array depends on the filter margins:
            - For a row shift, returned array size = up_margin + down_margin + 1
            - For a column shift, returned array size = left_margin + right_margin + 1

        :param fractional_shift: positive fractional shift of the subpixel position to be interpolated
        :type fractional_shift: float
        :return: a array of interpolator coefficients whose size depends on the filter margins
        :rtype: np.ndarray
        """

    @staticmethod
    def apply(resampling_area: np.ndarray, row_coeff: np.ndarray, col_coeff: np.ndarray) -> float:
        """
        Returns the value of the interpolated position

        :param resampling_area: area on which interpolator coefficients will be applied
        :type resampling_area: np.ndarray
        :param row_coeff: interpolator coefficients in rows
        :type row_coeff: np.ndarray
        :param col_coeff: interpolator coefficients in columns
        :type col_coeff: np.ndarray
        :return: the interpolated value of the position corresponding to col_coeff and row_coeff
        :rtype: float
        """

    def interpolate(
        self, image: np.ndarray, col_positions: np.ndarray, row_positions: np.ndarray, max_fractional_value: float = ...
    ) -> List:
        """
        Returns the values of the 8 interpolated position

        :param image: image
        :type image: np.ndarray
        :param positions: subpix positions to be interpolated
        :type positions: Tuple[np.ndarray, np.ndarray]
        :param max_fractional_value: maximum fractional value used to get coefficients
        :type max_fractional_value: float
        :return: the interpolated values of the corresponding subpix positions
        :rtype: List of float
        """

    def get_margins(self) -> Margins:
        """Returns filter's margins."""

class Bicubic(AbstractFilter):
    """Implementation of the Bicubic filter.

    With `alpha = -0.5` and a size of 4.
    """

class CardinalSine(AbstractFilter):
    """
    Implementation of the Normalized Cardinal Sine filter.
    """

    size: int

    def __init__(self, half_size: int = 6, fractional_shift: float = 0.25) -> None:
        """

        :param half_size: half filter size.
        :type half_size: int
        :param fractional_shift: interval between each interpolated point, sometimes referred to as precision.
                                 Expected value in the range [0,1[.
        :type fractional_shift: float
        """
