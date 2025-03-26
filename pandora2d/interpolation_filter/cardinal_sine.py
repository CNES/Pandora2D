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


@AbstractFilter.register_subclass("sinc_python")
class CardinalSinePython(AbstractFilter):
    """Implementation of the Normalized Cardinal Sine filter."""

    schema = {"method": "sinc_python", OptionalKey("size"): And(int, lambda a: 6 <= a <= 21)}

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
        self._HALF_SIZE = self.cfg.get("size", 6)  # pylint:disable=invalid-name
        self._SIZE = 1 + self._HALF_SIZE * 2  # pylint:disable=invalid-name
        self.fractional_shifts = np.arange(0, 1, fractional_shift)
        self.coeffs = compute_coefficient_table(filter_size=self._HALF_SIZE, fractional_shifts=self.fractional_shifts)

    @staticmethod
    def _check_fractional_shift(fractional_shift: float) -> None:
        if not 0 <= fractional_shift < 1:
            raise ValueError(f"fractional_shift greater than 0 and lower than 1 expected, got {fractional_shift}")

    @property
    def margins(self):
        """Return filter's Margins."""
        return Margins(self._HALF_SIZE, self._HALF_SIZE, self._HALF_SIZE, self._HALF_SIZE)

    def get_coeffs(self, fractional_shift: float) -> NDArray[np.floating]:
        index = self.fractional_shifts.searchsorted(fractional_shift)
        return self.coeffs[index]


def compute_coefficient_table(filter_size: int, fractional_shifts: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute normalized cardinal sine coefficients windowed by a Gaussian.

    Will compute the `2 * filter_size + 1` coefficients for each given fractional_shift in `fractional_shifts` and
    store them in the returned NDArray where:

        - Each row corresponds to a specific fractional shift value.
        - Each column corresponds to a coefficient at a specific position.

    The Gaussian window width correspond to the size of the filter.

    :param filter_size: Half number of coefficients to compute.
    :type filter_size: int
    :param fractional_shifts: At which fractional shifts to compute coefficients
    :type fractional_shifts: NDArray[np.floating]
    :return: 2D array with computed coefficients
    :rtype: NDArray[np.floating]
    """
    sigma = filter_size
    aux1 = (-2.0 * np.pi) / (sigma * sigma)
    coeff_range = np.arange(-filter_size, filter_size + 1)
    # The np.meshgrid function creates a grid of indices corresponding to the positions of the coefficients. It
    # generates two 2D arrays (xv and yv) where each element represents a combination of indices. In this case,
    # xv contains the indices of the coefficients, and yv contains the fractional shift values
    xv, yv = np.meshgrid(coeff_range, fractional_shifts, sparse=True)

    # (yv- xv) gives:
    # array([[ 6.  ,  5.  ,  4.  ,  3.  ,  2.  ,  1.  ,  0.  , -1.  , -2.  ,
    #         -3.  , -4.  , -5.  , -6.  ],
    #        [ 6.25,  5.25,  4.25,  3.25,  2.25,  1.25,  0.25, -0.75, -1.75,
    #         -2.75, -3.75, -4.75, -5.75],
    #        [ 6.5 ,  5.5 ,  4.5 ,  3.5 ,  2.5 ,  1.5 ,  0.5 , -0.5 , -1.5 ,
    #         -2.5 , -3.5 , -4.5 , -5.5 ],
    #        [ 6.75,  5.75,  4.75,  3.75,  2.75,  1.75,  0.75, -0.25, -1.25,
    #         -2.25, -3.25, -4.25, -5.25]])
    aux = yv - xv
    tab_coeffs = np.sinc(aux) * np.exp(aux1 * aux * aux)
    return tab_coeffs / np.nansum(tab_coeffs, axis=1, keepdims=True)
