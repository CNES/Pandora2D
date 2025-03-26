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
"""This module contains bicubic interpolation filter."""

from functools import lru_cache

import numpy as np
from pandora.margins import Margins

from .interpolation_filter import AbstractFilter


@AbstractFilter.register_subclass("bicubic_python")
class BicubicPython(AbstractFilter):
    """Implementation of the Bicubic filter.

    With `alpha = -0.5` and a size of 4.
    """

    _ALPHA = -0.5
    _SIZE = 4

    def __init__(self, cfg, **_):
        """
        Initialize a BicubicPython instance.

        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        self.schema = {"method": "bicubic_python"}
        super().__init__(cfg)

    @property
    def margins(self):
        """Return filter's Margins."""
        return Margins(1, 1, 2, 2)

    @lru_cache
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
        tab_coeffs = np.empty(4)
        alpha = self._ALPHA

        for i in range(4):
            dist = abs(-1.0 + i - fractional_shift)

            if dist <= 1.0:
                tab_coeffs[i] = (((alpha + 2.0) * dist - (alpha + 3.0)) * dist * dist) + 1.0
            else:
                tab_coeffs[i] = (((alpha * dist - 5.0 * alpha) * dist) + 8.0 * alpha) * dist - 4.0 * alpha

        return tab_coeffs
