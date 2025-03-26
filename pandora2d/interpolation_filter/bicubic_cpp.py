# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
"""This module contains cpp bicubic interpolation filter."""

import numpy as np

from pandora.margins import Margins

from .interpolation_filter import AbstractFilter
from ..interpolation_filter_cpp import interpolation_filter_bind


@AbstractFilter.register_subclass("bicubic")
class Bicubic(AbstractFilter):
    """
    Implementation of the Bicubic filter in cpp.

    With `alpha = -0.5` and a size of 4.
    """

    def __init__(self, cfg, **_):
        """
        Initialize a cpp Bicubic instance.

        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        self.schema = {"method": "bicubic"}
        super().__init__(cfg)
        self.cpp_instance = interpolation_filter_bind.Bicubic()

    @property
    def margins(self):
        """Return filter's Margins."""

        cpp_margins = self.cpp_instance.get_margins()

        return Margins(cpp_margins.left, cpp_margins.up, cpp_margins.right, cpp_margins.down)

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

        tab_coeffs = self.cpp_instance.get_coeffs(fractional_shift)

        return tab_coeffs
