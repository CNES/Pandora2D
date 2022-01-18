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
This module contains functions associated to the matching cost computation step.
"""

from typing import Dict
from json_checker import And, Checker

import xarray as xr
import numpy as np

from pandora import matching_cost

from pandora2d import img_tools


class MatchingCost:
    """
    Matching Cost class
    """

    _WINDOW_SIZE = 5

    def __init__(self, **cfg: str) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg["window_size"]
        self._matching_cost_method = self.cfg["matching_cost_method"]

    def check_conf(self, **cfg: str) -> Dict[str, str]:
        """
        Check the matching cost configuration

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE  # type: ignore

        schema = {
            "matching_cost_method": And(str, lambda mc: mc in ["ssd", "sad", "zncc"]),
            "window_size": And(int, lambda ws: ws > 0, lambda ws: ws % 2 != 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def allocate_cost_volumes(
        cost_volume_attr: dict,
        row: np.array,
        col: np.array,
        disp_min_col: int,
        disp_max_col: int,
        disp_min_row: int,
        disp_max_row: int,
        np_data: np.ndarray = None,
    ) -> xr.Dataset:
        """
        Allocate the cost volumes

        :param cost_volume_attr: the cost_volume's attributs product by Pandora
        :type cost_volume: xr.Dataset
        :param row: dimension of the image (row)
        :type row: np.array
        :param col: dimension of the image (columns)
        :type col: np.array
        :param disp_min_col: minimum disparity in columns
        :type disp_min_col: int
        :param disp_max_col: maximum disparity in columns
        :type disp_max_col: int
        :param disp_min_row: minimum disparity in lines
        :type disp_min_row: int
        :param disp_max_row: maximum disparity in lines
        :type disp_max_row: int
        :param np_data: 4D numpy.array og cost_volumes. Defaults to None.
        :type np_data: np.ndarray
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """
        disparity_range_col = np.arange(disp_min_col, disp_max_col + 1)
        disparity_range_row = np.arange(disp_min_row, disp_max_row + 1)

        # Create the cost volume
        if np_data is None:
            np_data = np.zeros(
                (len(row), len(col), len(disparity_range_col), len(disparity_range_row)), dtype=np.float32
            )

        cost_volumes = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], np_data)},
            coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
        )

        cost_volumes.attrs = cost_volume_attr

        return cost_volumes

    def compute_cost_volumes(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int,
        **cfg: Dict[str, dict]
    ) -> xr.Dataset:
        """

        Computes the cost volumes

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param min_col: minimum disparity in columns
        :type min_col: int
        :param max_col: maximum disparity in columns
        :type max_col: int
        :param min_row: minimum disparity in lines
        :type min_row: int
        :param max_row: maximum disparity in lines
        :type max_row: int
        :param cfg: matching_cost computation configuration
        :type max_row: dict
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        cost_volumes = xr.Dataset()
        # Initialize Pandora matching cost
        pandora_matching_cost_ = matching_cost.AbstractMatchingCost(**cfg)
        # Array with all y disparities
        disps_row = range(min_row, max_row + 1)
        for idx, disp_row in enumerate(disps_row):
            # Shift image in the y axis
            img_right_shift = img_tools.shift_img_pandora2d(img_right, disp_row)
            # Compute cost volume
            cost_volume = pandora_matching_cost_.compute_cost_volume(img_left, img_right_shift, min_col, max_col)
            # Mask cost volume
            pandora_matching_cost_.cv_masked(img_left, img_right_shift, cost_volume, min_col, max_col)
            # If first iteration, initialize cost_volumes dataset
            if idx == 0:
                c_row = cost_volume["cost_volume"].coords["row"]
                c_col = cost_volume["cost_volume"].coords["col"]

                # First pixel in the image that is fully computable (aggregation windows are complete)
                row = np.arange(c_row[0], c_row[-1] + 1)
                col = np.arange(c_col[0], c_col[-1] + 1)

                cost_volumes = self.allocate_cost_volumes(
                    cost_volume.attrs, row, col, min_col, max_col, min_row, max_row, None
                )
            # Add current cost volume to the cost_volumes dataset
            cost_volumes["cost_volumes"][:, :, :, idx] = cost_volume["cost_volume"].data

        return cost_volumes
