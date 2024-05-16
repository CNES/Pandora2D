# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
from typing import Dict
from abc import abstractmethod, ABC

import numpy as np

from pandora.margins.descriptors import NullMargins


class AbstractFilter(ABC):
    """
    Abstract Filter class
    """

    interpolation_filter_methods_avail: Dict = {}
    _interpolation_filter_method = None
    cfg = None
    margins = NullMargins()

    def __new__(cls, filter_method: str | None = None):
        """
        Return the plugin associated with the interpolation filter given in the configuration

        :param filter_method: filter_method
        :type cfg: str | None
        """

        if cls is AbstractFilter:
            if isinstance(filter_method, str):
                try:
                    return super(AbstractFilter, cls).__new__(cls.interpolation_filter_methods_avail[filter_method])
                except KeyError:
                    logging.error("No subpixel method named %s supported", filter_method)
                    raise KeyError
        return super(AbstractFilter, cls).__new__(cls)

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
