#!/usr/bin/env python
#
# Copyright (c) 2025 CS GROUP France
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
This module contains functions associated to the phase cross correlation method used in the estimation step.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from json_checker import And, Checker
from skimage.registration import phase_cross_correlation  # pylint: disable=no-name-in-module

from . import estimation


@estimation.AbstractEstimation.register_subclass("phase_cross_correlation")
class PhaseCrossCorrelation(estimation.AbstractEstimation):
    """
    PhaseCrossCorrelation class allows to perform estimation
    """

    # Default configuration, do not change these values
    _RANGE_COL = 5
    _RANGE_ROW = 5
    _SAMPLE_FACTOR = 1

    def __init__(self, cfg: Dict) -> None:
        """
        :param cfg: optional configuration, {'range_col': int, 'range_row': int, 'sample_factor': int}
        :type cfg: dict
        :return: None
        """

        self.cfg = self.check_conf(cfg)
        self._range_col = self.cfg["range_col"]
        self._range_row = self.cfg["range_row"]
        self._sample_factor = self.cfg["sample_factor"]
        self._estimation_method = self.cfg["estimation_method"]

    def check_conf(self, cfg: Dict) -> Dict:
        """
        Check the estimation configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        # Give the default value if the required element is not in the conf
        if "range_col" not in cfg:
            cfg["range_col"] = self._RANGE_COL

        if "range_row" not in cfg:
            cfg["range_row"] = self._RANGE_ROW

        if "sample_factor" not in cfg:
            cfg["sample_factor"] = self._SAMPLE_FACTOR

        # Estimation schema config
        schema = {
            "estimation_method": And(str, lambda estimation_method: estimation_method in ["phase_cross_correlation"]),
            "range_row": And(int, lambda range_row: range_row > 2, lambda range_row: range_row % 2 != 0),
            "range_col": And(int, lambda range_col: range_col > 2, lambda range_col: range_col % 2 != 0),
            "sample_factor": And(int, lambda sf: sf % 10 == 0 or sf == 1, lambda sf: sf > 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    def compute_estimation(self, img_left: xr.Dataset, img_right: xr.Dataset) -> Tuple[Dict, Dict, np.ndarray, dict]:
        """
        Compute the phase cross correlation method

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :return: row disparity: list
                col disparity: list
                Calculated shifts: np.ndarray
                Extra information about estimation: dict
        :rtype: list, list, np.ndarray, dict
        """

        # https://scikit-image.org/docs/stable/api/
        # skimage.registration.html#skimage.registration.phase_cross_correlation
        shifts, error, phasediff = phase_cross_correlation(
            img_left["im"].data, img_right["im"].data, upsample_factor=self._sample_factor
        )

        # reformat outputs
        phasediff = "{:.{}e}".format(phasediff, 8)
        # -shifts because of pandora2d convention
        row_disparity = {"init": round(-shifts[0]), "range": int(self._range_row)}
        col_disparity = {"init": round(-shifts[1]), "range": int(self._range_col)}

        logging.info("Estimation result is %s in columns and %s in row", -shifts[1], -shifts[0])
        logging.debug("Translation invariant normalized RMS error between left and right is %s", error)
        logging.debug("Global phase difference between the two images is %s", phasediff)

        extra_dict = {"error": error, "phase_diff": phasediff}

        return row_disparity, col_disparity, -shifts, extra_dict
