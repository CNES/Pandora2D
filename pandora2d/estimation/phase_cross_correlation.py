#!/usr/bin/env python
# coding: utf8
#
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
This module contains functions associated to the SURF method used in the estimation step.
"""

from math import ceil, floor
import logging
from typing import Dict, Tuple
from json_checker import And, Checker
import xarray as xr
from skimage.registration import phase_cross_correlation
from . import estimation


@estimation.AbstractEstimation.register_subclass("phase_cross_correlation")
class PhaseCrossCorrelation(estimation.AbstractEstimation):
    """
    PhaseCrossCorrelation class allows to perform estimation
    """

    _RANGE_COL = 5
    _RANGE_ROW = 5

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._range_col = self.cfg["range_col"]
        self._range_row = self.cfg["range_row"]

    def check_conf(self, **cfg: str) -> Dict[str, str]:
        """
        Check the estimation configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        if "range_col" not in cfg:
            cfg["range_col"] = self._RANGE_COL  # type: ignore

        if "range_row" not in cfg:
            cfg["range_row"] = self._RANGE_ROW  # type: ignore

        schema = {
            "estimation_method": And(str, lambda x: x in ["phase_cross_correlation"]),
            "range_col": int,
            "range_row": int,
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def ceil_or_floor(number: float) -> int:
        """
        Upper round a number
        :param number: number to round
        :type number: float
        :return: rounded
        :rtype: int
        """

        return ceil(number) if number > 0 else floor(number)

    def estimation_method(
            self, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        """
        Compute the phase cross correlation method

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
                 shifted: global shifts between left and right
        :rtype: min_col: int
                max_col: int
                min_row: int
                max_row: int
                shifts: xr.dataArray
        """

        left = img_left["im"].data
        right = img_right["im"].data

        shifted, _, _ = phase_cross_correlation(left, right, upsample_factor=100)

        min_col = -shifted[1] - int(self._range_col / 2)  # type: ignore
        max_col = -shifted[1] + int(self._range_col / 2)  # type: ignore
        min_row = -shifted[0] - int(self._range_row / 2)  # type: ignore
        max_row = -shifted[0] + int(self._range_row / 2)  # type: ignore

        logging.info("Estimation result is %s in columns and %s in row", -shifted[1], -shifted[0])

        outputs = (self.ceil_or_floor(min_col),
                   self.ceil_or_floor(max_col),
                   self.ceil_or_floor(min_row),
                   self.ceil_or_floor(max_row),
                   -shifted)

        return outputs