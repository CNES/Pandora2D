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
"""
This module contains functions associated to the interpolation filters.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, List
from abc import abstractmethod, ABC

import math
import numpy as np
from json_checker import Checker

from pandora.margins.descriptors import NullMargins


class AbstractFilter(ABC):
    """
    Abstract Filter class
    """

    interpolation_filter_methods_avail: Dict = {}
    _interpolation_filter_method = None
    margins = NullMargins()
    _SIZE = 4
    cpp_instance = None

    def __new__(cls, cfg: dict | None = None, **kwargs):
        """
        Return the plugin associated with the interpolation filter given in the configuration

        :param filter_method: filter_method
        :type cfg: str | None
        """

        if cls is AbstractFilter:
            if isinstance(cfg["method"], str):
                filter_method = cfg["method"]
                try:
                    return super(AbstractFilter, cls).__new__(cls.interpolation_filter_methods_avail[filter_method])
                except KeyError:
                    logging.error("No subpixel method named %s supported", filter_method)
                    raise KeyError
        return super(AbstractFilter, cls).__new__(cls)

    def __init__(self, cfg: Dict, **_) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(cfg)

    def check_conf(self, cfg: Dict) -> Dict:
        """
        Check the refinement method configuration.

        :param cfg: user_config for refinement method
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
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
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
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

        return (row_coeff.dot(resampling_area)).dot(col_coeff)

    def interpolate(
        self, image: np.ndarray, positions: Tuple[np.ndarray, np.ndarray], max_fractional_value: float = 0.998046875
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
