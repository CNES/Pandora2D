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
import copy
from typing import Dict, List, cast, Union
from json_checker import And, Checker

import xarray as xr
import numpy as np

from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.margins.descriptors import HalfWindowMargins


from pandora2d import img_tools


class MatchingCost:
    """
    Matching Cost class
    """

    _WINDOW_SIZE = 5
    _STEP = [1, 1]
    margins = HalfWindowMargins()

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(cfg)
        self._window_size = int(self.cfg["window_size"])
        self._matching_cost_method = self.cfg["matching_cost_method"]
        # Cast to int in order to help mypy because self.cfg is a Dict and it can not know the type of step.
        self._step_row = cast(int, self.cfg["step"][0])
        self._step_col = cast(int, self.cfg["step"][1])

        # Init pandora items
        self.pandora_matching_cost_: Union[matching_cost.AbstractMatchingCost, None] = None
        self.grid_: xr.Dataset = None

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """
        Check the matching cost configuration

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE
        if "step" not in cfg:
            cfg["step"] = self._STEP

        schema = {
            "matching_cost_method": And(str, lambda mc: mc in ["ssd", "sad", "zncc"]),
            "window_size": And(int, lambda ws: ws > 0, lambda ws: ws % 2 != 0),
            "step": And(list, lambda x: len(x) == 2, lambda y: all(val >= 1 for val in y)),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def allocate_cost_volumes(
        cost_volume_attr: dict,
        row: np.ndarray,
        col: np.ndarray,
        col_disparity: List[int],
        row_disparity: List[int],
        np_data: np.ndarray = None,
    ) -> xr.Dataset:
        """
        Allocate the cost volumes

        :param cost_volume_attr: the cost_volume's attributs product by Pandora
        :type cost_volume: xr.Dataset
        :param row: dimension of the image (row)
        :type row: np.ndarray
        :param col: dimension of the image (columns)
        :type col: np.ndarray
        :param col_disparity: min and max disparities for columns.
        :type col_disparity: List[int]
        :param row_disparity: min and max disparities for rows.
        :type row_disparity: List[int]
        :param np_data: 4D numpy.ndarray og cost_volumes. Defaults to None.
        :type np_data: np.ndarray
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        disp_min_col, disp_max_col = col_disparity
        disp_min_row, disp_max_row = row_disparity

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

        # del pandora attributes
        del cost_volumes.attrs["col_to_compute"]
        del cost_volumes.attrs["sampling_interval"]

        return cost_volumes

    def allocate_cost_volume_pandora(
        self, img_left: xr.Dataset, img_right: xr.Dataset, grid_min_col: np.ndarray, grid_max_col: np.ndarray, cfg: Dict
    ) -> None:
        """

        Allocate the cost volume for pandora

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param grid_min_col: grid containing min disparities for columns.
        :type grid_min_col: np.ndarray
        :param grid_max_col: grid containing max disparities for columns.
        :type grid_max_col: np.ndarray
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :return: None
        """
        # Adapt Pandora matching cost configuration
        copy_matching_cost_cfg_with_step = copy.deepcopy(cfg)
        copy_matching_cost_cfg_with_step["step"] = self._step_col
        img_left.attrs["disparity_source"] = img_left.attrs["col_disparity_source"]

        # Initialize Pandora matching cost
        self.pandora_matching_cost_ = matching_cost.AbstractMatchingCost(**copy_matching_cost_cfg_with_step)

        # Initialize pandora an empty grid for cost volume
        self.grid_ = self.pandora_matching_cost_.allocate_cost_volume(img_left, (grid_min_col, grid_max_col), cfg)

        # Compute validity mask to identify invalid points in cost volume
        self.grid_ = validity_mask(img_left, img_right, self.grid_)

    def compute_cost_volumes(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        grid_min_col: np.ndarray,
        grid_max_col: np.ndarray,
        grid_min_row: np.ndarray,
        grid_max_row: np.ndarray,
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
        :param grid_min_col: grid containing min disparities for columns.
        :type grid_min_col: np.ndarray
        :param grid_max_col: grid containing max disparities for columns.
        :type grid_max_col: np.ndarray
        :param grid_min_row: grid containing min disparities for rows.
        :type grid_min_row: np.ndarray
        :param grid_max_row: grid containing max disparities for rows.
        :type grid_max_row: np.ndarray
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        cost_volumes = xr.Dataset()

        # Adapt Pandora matching cost configuration
        img_left.attrs["disparity_source"] = img_left.attrs["col_disparity_source"]

        # Obtain absolute min and max disparities
        min_row, max_row = self.pandora_matching_cost_.get_min_max_from_grid(grid_min_row, grid_max_row)
        min_col, max_col = self.pandora_matching_cost_.get_min_max_from_grid(grid_min_col, grid_max_col)

        # Array with all y disparities
        disps_row = range(min_row, max_row + 1)
        row_step = None
        for idx, disp_row in enumerate(disps_row):
            # Shift image in the y axis
            img_right_shift = img_tools.shift_img_pandora2d(img_right, disp_row)
            # Compute cost volume
            cost_volume = self.pandora_matching_cost_.compute_cost_volume(img_left, img_right_shift, self.grid_)
            # Mask cost volume
            self.pandora_matching_cost_.cv_masked(img_left, img_right_shift, cost_volume, grid_min_col, grid_max_col)
            # If first iteration, initialize cost_volumes dataset
            if idx == 0:
                c_row = cost_volume["cost_volume"].coords["row"]
                c_col = cost_volume["cost_volume"].coords["col"]

                # First pixel in the image that is fully computable (aggregation windows are complete)
                row = np.arange(c_row[0], c_row[-1] + 1, self._step_row)
                col = np.arange(c_col[0], c_col[-1] + 1, self._step_col)

                cost_volumes = self.allocate_cost_volumes(
                    cost_volume.attrs, row, col, [min_col, max_col], [min_row, max_row], None
                )

                # Number of line to be taken as a function of the step.
                # Note that the row vector may not start at zero.
                row_step = np.arange(0, c_row[-1] + 1 - c_row[0], self._step_row)

            # Add current cost volume to the cost_volumes dataset
            cost_volumes["cost_volumes"].data[:, :, :, idx] = cost_volume["cost_volume"].data[row_step, :, :]

        # Add disparity source
        del cost_volumes.attrs["disparity_source"]
        cost_volumes.attrs["col_disparity_source"] = img_left.attrs["col_disparity_source"]
        cost_volumes.attrs["row_disparity_source"] = img_left.attrs["row_disparity_source"]

        return cost_volumes
