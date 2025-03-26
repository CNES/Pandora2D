#  Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This module contains cardinal sine interpolation filter."""

from __future__ import annotations

from typing import Dict

import numpy as np
from json_checker import And, OptionalKey
from numpy.typing import NDArray

from pandora.margins import Margins

from .interpolation_filter import AbstractFilter
from ..interpolation_filter_cpp import interpolation_filter_bind


@AbstractFilter.register_subclass("sinc")
class CardinalSine(AbstractFilter):
    """Implementation of the Normalized Cardinal Sine filter in C++."""

    schema = {"method": "sinc", OptionalKey("size"): And(int, lambda a: 6 <= a <= 21)}

    def __init__(self, cfg: Dict, fractional_shift: float = 0.5):
        """

        :param cfg: config
        :type cfg: dict
        :param fractional_shift: interval between each interpolated point, sometimes referred to as precision.
                                 Expected value in the range [0,1[.
        :type fractional_shift: float
        """
        super().__init__(cfg)
        self._check_fractional_shift(fractional_shift)
        half_size = self.cfg.get("size", 6)
        self.cpp_instance = interpolation_filter_bind.CardinalSine(half_size, fractional_shift)

    @staticmethod
    def _check_fractional_shift(fractional_shift: float) -> None:
        if not 0 <= fractional_shift < 1:
            raise ValueError(f"fractional_shift greater than 0 and lower than 1 expected, got {fractional_shift}")

    @property
    def _SIZE(self):  # pylint: disable=invalid-name
        """Return filter's size."""
        return self.cpp_instance.size

    @property
    def margins(self):
        """Return filter's Margins."""
        cpp_margins = self.cpp_instance.get_margins()

        return Margins(cpp_margins.left, cpp_margins.up, cpp_margins.right, cpp_margins.down)

    def get_coeffs(self, fractional_shift: float) -> NDArray[np.floating]:
        return self.cpp_instance.get_coeffs(fractional_shift)
