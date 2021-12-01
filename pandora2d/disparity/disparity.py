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
This module contains functions associated to the disparity map computation step.
"""

from typing import Dict, Union, Tuple
from json_checker import Or, And, Checker

import numpy as np
import xarray as xr
from pandora.check_json import update_conf


class Disparity:
    """
    Disparity class
    """

    _INVALID_DISPARITY = -9999

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        self.cfg = self.check_conf(**cfg)
        self._invalid_disparity = self.cfg["invalid_disparity"]

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Check the disparity configuration

        Returns:
            Dict[str, Union[str, int]]: disparity configuration
        """
        # Give the default value if the required element is not in the configuration
        if "invalid_disparity" not in cfg:
            cfg["invalid_disparity"] = self._INVALID_DISPARITY

        cfg = update_conf({"disparity_method": "wta"}, cfg)

        schema = {
            "disparity_method": And(str, lambda x: x in ["wta"]),
            "invalid_disparity": Or(int, float, lambda input: np.isnan(input), lambda input: np.isinf(input)),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def min_split(cost_volumes: xr.Dataset, axis: int) -> np.ndarray:
        """
        Find the indices of the minimum values for a 4D DataArray, along axis.
        Memory consumption is reduced by splitting the 4D Array.

        :param cost_volumes: the cost volume dataset
        :type cost_volumes: xarray.Dataset
        :param axis: research axis
        :type axis: int
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """

        cv_dims = cost_volumes["cost_volumes"].shape
        disps_min = np.zeros((cv_dims[0], cv_dims[1], cv_dims[5 - axis]), dtype=np.float32)

        # Numpy min is making a copy of the cost volume.
        # To reduce memory, numpy min is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_y = np.array_split(cost_volumes["cost_volumes"].data, np.arange(100, cv_dims[1], 100), axis=0)

        y_begin = 0

        for col, cv_y in enumerate(cv_chunked_y):  # pylint: disable=unused-variable
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_x = np.array_split(cv_y, np.arange(100, cv_dims[0], 100), axis=1)
            x_begin = 0
            for row, cv_x in enumerate(cv_chunked_x):  # pylint: disable=unused-variable
                disps_min[y_begin : y_begin + cv_y.shape[0], x_begin : x_begin + cv_x.shape[1], :] = np.min(
                    cv_x, axis=axis
                )
                x_begin += cv_x.shape[1]

            y_begin += cv_y.shape[0]

        return disps_min

    @staticmethod
    def max_split(cost_volumes: xr.Dataset, axis: int) -> np.ndarray:
        """
        Find the indices of the minimum values for a 4D DataArray, along axis.
        Memory consumption is reduced by splitting the 4D Array.

        :param cost_volumes: the cost volume dataset
        :type cost_volumes: xarray.Dataset
        :param axis: research axis
        :type axis: int
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """
        cv_dims = cost_volumes["cost_volumes"].shape
        disps_max = np.zeros((cv_dims[0], cv_dims[1], cv_dims[5 - axis]), dtype=np.float32)

        # Numpy min is making a copy of the cost volume.
        # To reduce memory, numpy min is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_y = np.array_split(cost_volumes["cost_volumes"].data, np.arange(100, cv_dims[1], 100), axis=0)

        y_begin = 0

        for col, cv_y in enumerate(cv_chunked_y):  # pylint: disable=unused-variable
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_x = np.array_split(cv_y, np.arange(100, cv_dims[0], 100), axis=1)
            x_begin = 0
            for row, cv_x in enumerate(cv_chunked_x):  # pylint: disable=unused-variable
                disps_max[y_begin : y_begin + cv_y.shape[0], x_begin : x_begin + cv_x.shape[1], :] = np.max(
                    cv_x, axis=axis
                )
                x_begin += cv_x.shape[1]

            y_begin += cv_y.shape[0]

        return disps_max

    @staticmethod
    def argmax_split(max_maps: np.array, axis: int) -> np.ndarray:
        """
        Find the indices of the minimum values for a 3D DataArray, along axis 2.
        Memory consumption is reduced by splitting the 3D Array.

        :param cost_volume: the cost volume dataset
        :type cost_volume: xarray.Dataset
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """
        ncol, nrow, ndsp = max_maps.shape  # pylint: disable=unused-variable
        disp = np.zeros((ncol, nrow), dtype=np.int)

        # Numpy argmin is making a copy of the cost volume.
        # To reduce memory, numpy argmin is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_y = np.array_split(max_maps, np.arange(100, ncol, 100), axis=0)

        y_begin = 0

        for col, cv_y in enumerate(cv_chunked_y):  # pylint: disable=unused-variable
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_x = np.array_split(cv_y, np.arange(100, nrow, 100), axis=1)
            x_begin = 0
            for row, cv_x in enumerate(cv_chunked_x):  # pylint: disable=unused-variable
                disp[y_begin : y_begin + cv_y.shape[0], x_begin : x_begin + cv_x.shape[1]] = np.argmax(cv_x, axis=axis)
                x_begin += cv_x.shape[1]

            y_begin += cv_y.shape[0]

        return disp

    @staticmethod
    def argmin_split(min_maps: np.array, axis: int) -> np.ndarray:
        """
        Find the indices of the minimum values for a 3D DataArray, along axis 2.
        Memory consumption is reduced by splitting the 3D Array.

        :param cost_volume: the cost volume dataset
        :type cost_volume: xarray.Dataset
        :return: the disparities for which the cost volume values are the smallest
        :rtype: np.ndarray
        """
        ncol, nrow, ndsp = min_maps.shape  # pylint: disable=unused-variable
        disp = np.zeros((ncol, nrow), dtype=np.int)

        # Numpy argmin is making a copy of the cost volume.
        # To reduce memory, numpy argmin is applied on a small part of the cost volume.
        # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
        cv_chunked_y = np.array_split(min_maps, np.arange(100, ncol, 100), axis=0)

        y_begin = 0

        for col, cv_y in enumerate(cv_chunked_y):  # pylint: disable=unused-variable
            # To reduce memory, the cost volume is split (along the col axis) into
            # multiple sub-arrays with a step of 100
            cv_chunked_x = np.array_split(cv_y, np.arange(100, nrow, 100), axis=1)
            x_begin = 0
            for row, cv_x in enumerate(cv_chunked_x):  # pylint: disable=unused-variable
                disp[y_begin : y_begin + cv_y.shape[0], x_begin : x_begin + cv_x.shape[1]] = np.argmin(cv_x, axis=axis)
                x_begin += cv_x.shape[1]

            y_begin += cv_y.shape[0]

        return disp

    def compute_disp_maps(self, cost_volumes: xr.Dataset) -> Tuple[np.array, np.array]:
        """
        Disparity computation by applying the Winner Takes All strategy

        :param cost_volumes: the cost volumes datsset with the data variables:
            - cost_volume 4D xarray.DataArray (row, col, disp_row, disp_col)
        :type cost_volumes: xr.Dataset
        :return: Two numpy.array:

                - disp_map_col : disparity map for columns
                - disp_map_row : disparity map for row
        :rtype: tuple (numpy.array, numpy.array)
        """

        indices_nan = np.isnan(cost_volumes["cost_volumes"].data)

        # Winner Takes All strategy
        if cost_volumes.attrs["type_measure"] == "max":
            cost_volumes["cost_volumes"].data[indices_nan] = -np.inf
            # -------compute disp_map row---------
            # process of maximum for dispx
            maps_max_x = self.max_split(cost_volumes, 2)
            # process of argmax for dispy
            disp_map_row = cost_volumes["disp_row"].data[self.argmax_split(maps_max_x, 2)]
            # -------compute disp_map col---------
            # process of maximum for dispy
            maps_max_y = self.max_split(cost_volumes, 3)
            # process of argmax for dispx
            disp_map_col = cost_volumes["disp_col"].data[self.argmax_split(maps_max_y, 2)]

        else:
            # -------compute disp_map row---------
            cost_volumes["cost_volumes"].data[indices_nan] = np.inf
            # process of minimum for dispx
            maps_min_x = self.min_split(cost_volumes, 2)
            # process of argmin for disp
            disp_map_row = cost_volumes["disp_row"].data[self.argmin_split(maps_min_x, 2)]
            # -------compute disp_map col---------
            # process of maximum for dispy
            maps_min_y = self.min_split(cost_volumes, 3)
            # process of argmin for dispx
            disp_map_col = cost_volumes["disp_col"].data[self.argmin_split(maps_min_y, 2)]

        invalid_mc = np.all(indices_nan, axis=(2, 3))
        disp_map_col = disp_map_col.astype("float32")
        disp_map_row = disp_map_row.astype("float32")
        disp_map_col[invalid_mc] = self._invalid_disparity
        disp_map_row[invalid_mc] = self._invalid_disparity

        return disp_map_col, disp_map_row
