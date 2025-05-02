#!/usr/bin/env python
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the disparity map computation step.
"""

from typing import Dict, Tuple, Callable
from json_checker import Or, And, Checker
import numpy as np
import xarray as xr

from pandora.margins.descriptors import NullMargins
from pandora.margins import Margins

from pandora2d.constants import Criteria


class Disparity:
    """
    Disparity class
    """

    _INVALID_DISPARITY = -9999
    margins = NullMargins()

    def __init__(self, cfg: Dict) -> None:
        self.cfg = self.check_conf(cfg)
        self._invalid_disparity = self.cfg["invalid_disparity"]

    def check_conf(self, cfg: Dict) -> Dict:
        """
        Check the disparity configuration

        Returns:
            Dict[str, Union[str, int, float]]: disparity configuration
        """
        # Give the default value if the required element is not in the configuration
        if "invalid_disparity" not in cfg:
            cfg["invalid_disparity"] = self._INVALID_DISPARITY
        elif cfg["invalid_disparity"] == "NaN":
            cfg["invalid_disparity"] = np.nan

        schema = {
            "disparity_method": And(str, lambda x: x in ["wta"]),
            "invalid_disparity": Or(int, float, lambda input: np.isnan(input), lambda input: np.isinf(input)),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def extrema_split(cost_volumes: xr.Dataset, axis: int, extrema_func: Callable) -> np.ndarray:
        """
        Find the indices of the minimum values for a 4D DataArray, along axis.
        Memory consumption is reduced by splitting the 4D Array.

        :param cost_volumes: the cost volume dataset
        :type cost_volumes: xarray.Dataset
        :param axis: research axis
        :type axis: int
        :param extrema_func: minimal or maximal research
        :type extrema_func: Callable
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """

        cv_dims = cost_volumes["cost_volumes"].shape
        disps = np.zeros((cv_dims[0], cv_dims[1], cv_dims[5 - axis]), dtype=np.float32)

        # Numpy min is making a copy of the cost volume.
        # To reduce memory, numpy min is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_row = np.array_split(cost_volumes["cost_volumes"].data, np.arange(100, cv_dims[1], 100), axis=0)

        row_begin = 0

        for _, cv_row in enumerate(cv_chunked_row):
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_col = np.array_split(cv_row, np.arange(100, cv_dims[0], 100), axis=1)
            col_begin = 0
            for _, cv_col in enumerate(cv_chunked_col):
                disps[row_begin : row_begin + cv_row.shape[0], col_begin : col_begin + cv_col.shape[1], :] = (
                    extrema_func(cv_col, axis=axis)
                )
                col_begin += cv_col.shape[1]

            row_begin += cv_row.shape[0]

        return disps

    @staticmethod
    def get_score(cost_volume: np.ndarray, extrema_func: Callable) -> np.ndarray:
        """
        Find the indicated extrema values for a 3D DataArray, along axis 2.
        Memory consumption is reduced by splitting the 3D Array.

        :param cost_volume: the cost volume dataset
        :type cost_volume: xarray.Dataset
        :param extrema_func: minimal or maximal research
        :type extrema_func: Callable
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """

        ncol, nrow, _ = cost_volume.shape
        score = np.empty((ncol, nrow), dtype=np.float32)

        # Numpy argmin is making a copy of the cost volume.
        # To reduce memory, numpy argmin is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100

        cv_chunked_y = np.array_split(cost_volume.data, np.arange(100, ncol, 100), axis=0)

        y_begin = 0

        for _, cv_y in enumerate(cv_chunked_y):
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_x = np.array_split(cv_y, np.arange(100, nrow, 100), axis=1)
            x_begin = 0
            for _, cv_x in enumerate(cv_chunked_x):
                score[y_begin : y_begin + cv_y.shape[0], x_begin : x_begin + cv_x.shape[1]] = extrema_func(cv_x, axis=2)
                x_begin += cv_x.shape[1]

            y_begin += cv_y.shape[0]

        return score

    @staticmethod
    def arg_split(maps: np.ndarray, axis: int, extrema_func: Callable) -> np.ndarray:
        """
        Find the indices of the maximum values for a 3D DataArray, along axis.
        Memory consumption is reduced by splitting the 3D Array.

        :param maps: maps with maximum
        :type maps: np.ndarray
        :param axis: axis
        :type axis: int
        :param extrema_func: minimal or maximal index research
        :type extrema_func: Callable
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """
        ncol, nrow, _ = maps.shape
        disp = np.zeros((ncol, nrow), dtype=np.int64)

        # Numpy argmin is making a copy of the cost volume.
        # To reduce memory, numpy argmin is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_row = np.array_split(maps, np.arange(100, ncol, 100), axis=0)

        row_begin = 0

        for _, cv_row in enumerate(cv_chunked_row):
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_col = np.array_split(cv_row, np.arange(100, nrow, 100), axis=1)
            col_begin = 0
            for _, cv_col in enumerate(cv_chunked_col):
                disp[row_begin : row_begin + cv_row.shape[0], col_begin : col_begin + cv_col.shape[1]] = extrema_func(
                    cv_col, axis=axis
                )
                col_begin += cv_col.shape[1]

            row_begin += cv_row.shape[0]

        return disp

    def compute_disp_maps(self, cost_volumes: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Disparity computation by applying the Winner Takes All strategy

        :param cost_volumes: the cost volumes dataset with the data variables:
            - cost_volume 4D xarray.DataArray (row, col, disp_row, disp_col)
        :type cost_volumes: xr.Dataset
        :return: three numpy.array:

                - disp_map_col : disparity map for columns
                - disp_map_row : disparity map for row
                - correlation_score_map : map containing matching_cost step score
        :rtype: tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """

        disparity_margins = cost_volumes.attrs["disparity_margins"]

        # Check margins presence
        if disparity_margins is not None and disparity_margins != Margins(0, 0, 0, 0):
            margins = disparity_margins.asdict()
            for key in margins.keys():
                margins[key] *= cost_volumes.attrs["subpixel"]

            cost_volumes_user = cost_volumes.isel(
                disp_row=slice(margins["up"], -margins["down"] or None),
                disp_col=slice(margins["left"], -margins["right"] or None),
            )

        else:
            cost_volumes_user = cost_volumes.copy(deep=True)

        invalid_index = cost_volumes_user["criteria"].data != Criteria.VALID

        # Winner Takes All strategy
        if cost_volumes.attrs["type_measure"] == "max":

            cost_volumes_user["cost_volumes"].data[invalid_index] = -np.inf
            # -------compute disp_map row---------
            # process of maximum for dispx
            maps_max_col = self.extrema_split(cost_volumes_user, 3, np.max)
            # process of argmax for dispy
            disp_map_row = cost_volumes_user["disp_row"].data[self.arg_split(maps_max_col, 2, np.argmax)]
            # -------compute disp_map col---------
            # process of maximum for dispy
            maps_max_row = self.extrema_split(cost_volumes_user, 2, np.max)
            # process of argmax for dispx
            disp_map_col = cost_volumes_user["disp_col"].data[self.arg_split(maps_max_row, 2, np.argmax)]
            # --------compute correlation score----
            score_map = self.get_score(maps_max_row, np.max)

        else:
            # -------compute disp_map row---------
            cost_volumes_user["cost_volumes"].data[invalid_index] = np.inf
            # process of minimum for dispx
            maps_min_col = self.extrema_split(cost_volumes_user, 3, np.min)
            # process of argmin for disp
            disp_map_row = cost_volumes_user["disp_row"].data[self.arg_split(maps_min_col, 2, np.argmin)]
            # -------compute disp_map col---------
            # process of maximum for dispy
            maps_min_row = self.extrema_split(cost_volumes_user, 2, np.min)
            # process of argmin for dispx
            disp_map_col = cost_volumes_user["disp_col"].data[self.arg_split(maps_min_row, 2, np.argmin)]
            # --------compute correlation score----
            score_map = self.get_score(maps_min_row, np.min)

        invalid_mc = np.all(invalid_index, axis=(2, 3))
        cost_volumes_user["cost_volumes"].data[invalid_index] = np.nan

        if cost_volumes["cost_volumes"].data.dtype != disp_map_col.dtype:
            disp_map_col = disp_map_col.astype(cost_volumes["cost_volumes"].data.dtype)
            disp_map_row = disp_map_row.astype(cost_volumes["cost_volumes"].data.dtype)
            score_map = score_map.astype(cost_volumes["cost_volumes"].data.dtype)

        disp_map_col[invalid_mc] = self._invalid_disparity
        disp_map_row[invalid_mc] = self._invalid_disparity
        score_map[invalid_mc] = self._invalid_disparity

        return disp_map_col, disp_map_row, score_map
