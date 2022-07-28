#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2022 CS GROUP France
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
This module contains functions associated to the estimation computation step.
"""

import logging
from typing import Dict, Tuple
from abc import abstractmethod, ABCMeta

import xarray as xr


class AbstractEstimation:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    estimation_methods_avail: Dict = {}
    _estimation_method = None
    cfg = None

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the refinement_method given in the configuration

        :param cfg: configuration {'refinement_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractEstimation:
            if isinstance(cfg["estimation_method"], str):
                try:
                    return super(AbstractEstimation, cls).__new__(
                        cls.estimation_methods_avail[cfg["estimation_method"]]
                    )
                except KeyError:
                    logging.error("No subpixel method named % supported", cfg["estimation_method"])
                    raise KeyError
            else:
                if isinstance(cfg["estimation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractEstimation, cls).__new__(
                            cls.estimation_methods_avail[cfg["estimation_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No subpixel method named % supported",
                            cfg["estimation_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractEstimation, cls).__new__(cls)
        return None

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
            cls.estimation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def estimation_method(
        self, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        """
        Compute the estimation method

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :return: min_col: min of disparity range for columns
                 max_col: max of disparity range for columns
                 min_row: min of disparity range for rows
                 max_row: max of disparity range for rows
                 shifts: global shifts between left and right
        :rtype: min_col: float
                max_col: float
                min_row: float
                max_row: float
                shifts: xr.dataArray
        """
