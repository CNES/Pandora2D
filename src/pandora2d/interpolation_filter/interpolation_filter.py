# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
"""
This module contains functions associated to the interpolation filters.
"""

from __future__ import annotations

import logging
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from json_checker import Checker

from pandora2d.margins import NullMargins

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class AbstractFilter(ABC):
    """
    Abstract Filter class
    """

    interpolation_filter_methods_avail: dict = {}
    _interpolation_filter_method = None
    _SIZE = 4
    cpp_instance = None

    def __new__(cls, cfg: dict | None = None, **kwargs):
        """
        Return the plugin associated with the interpolation filter given in the configuration

        :param filter_method: filter_method
        """

        if cls is AbstractFilter:
            if isinstance(cfg["method"], str):
                filter_method = cfg["method"]
                try:
                    return super().__new__(cls.interpolation_filter_methods_avail[filter_method])
                except KeyError:
                    logging.error("No subpixel method named %s supported", filter_method)
                    raise KeyError
        return super().__new__(cls)

    def __init__(self, cfg: dict, **_) -> None:
        """
        :param cfg: optional configuration, {}
        :return: None
        """
        self.cfg = self.check_conf(cfg)

    @property
    def margins(self):
        """Return filter's Margins."""
        return NullMargins()

    def check_conf(self, cfg: dict) -> dict:
        """
        Check the refinement method configuration.

        :param cfg: user_config for refinement method
        :return: cfg: global configuration
        """
        checker = Checker(self.schema)  # type: ignore[attr-defined]
        checker.validate(cfg)

        return cfg

    def desc(self) -> None:
        """
        Describes the interpolation filter
        :return: None
        """
        print(f"{self. _interpolation_filter_method} interpolation filter")

    @classmethod
    def register_subclass(cls, short_name: str) -> Callable[[type[Self]], type[Self]]:
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        """

        def decorator(subclass: type[Self]) -> type[Self]:
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            """
            cls.interpolation_filter_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def get_coeffs(self, fractional_shift: float) -> np.ndarray:
        """
        Returns the interpolator coefficients to be applied to the resampling area.

        The size of the returned array depends on the filter margins:
            - For a row shift, returned array size = up_margin + down_margin + 1
            - For a column shift, returned array size = left_margin + right_margin + 1

        :param fractional_shift: positive fractional shift of the subpixel position to be interpolated
        :return: a array of interpolator coefficients whose size depends on the filter margins
        """

    @staticmethod
    def apply(resampling_area: np.ndarray, row_coeff: np.ndarray, col_coeff: np.ndarray) -> float:
        """
        Returns the value of the interpolated position

        :param resampling_area: area on which interpolator coefficients will be applied
        :param row_coeff: interpolator coefficients in rows
        :param col_coeff: interpolator coefficients in columns
        :return: the interpolated value of the position corresponding to col_coeff and row_coeff
        """

        return (row_coeff.dot(resampling_area)).dot(col_coeff)

    def interpolate(
        self, image: np.ndarray, positions: tuple[np.ndarray, np.ndarray], max_fractional_value: float = 0.998046875
    ) -> list:
        """
        Returns the values of the 8 interpolated position

        :param image: image
        :param positions: subpix positions to be interpolated
        :param max_fractional_value: maximum fractional value used to get coefficients
        :return: the interpolated values of the corresponding subpix positions
        """

        # Initialisation of the result list
        interpolated_positions = []

        # Epsilon machine
        eps = np.finfo(np.float32).eps

        for pos_col, pos_row in zip(*positions):
            # get_coeffs method receives positive coefficients
            fractional_row = abs(math.modf(pos_row)[0])
            fractional_col = abs(math.modf(pos_col)[0])

            # If the subpixel shift is too close to 1, max_fractional_value is returned to avoid rounding.
            if 1 - fractional_row < eps:
                fractional_row = max_fractional_value
            if 1 - fractional_col < eps:
                fractional_col = max_fractional_value

            # Get interpolation coefficients for fractional_row and fractional_col shifts
            coeffs_row = self.get_coeffs(fractional_row)
            coeffs_col = self.get_coeffs(fractional_col)

            # Computation of the top left point of the resampling area
            # on which the interpolating coefficients will be applied with apply method
            # In cost_surface, row dimension is disp_col and column dimension is disp_row,
            # then we use margins.left for row and margins.up for col
            top_left_area_row = int(np.floor(pos_row) - self.margins.left)
            top_left_area_col = int(np.floor(pos_col) - self.margins.up)

            # Resampling area to which we will apply the interpolator coefficients
            resampling_area = image[
                top_left_area_row : top_left_area_row + self._SIZE, top_left_area_col : top_left_area_col + self._SIZE
            ]

            # Application of the interpolator coefficients on resampling area
            interpolated_value = self.apply(resampling_area, coeffs_row, coeffs_col)

            # Add new interpolated value in the result list
            interpolated_positions.append(interpolated_value)

        return interpolated_positions
