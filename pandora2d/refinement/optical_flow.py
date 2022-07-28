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
This module contains functions associated to the optical flow method used in the refinement step.
"""

from typing import Dict, Tuple
from json_checker import And, Or, Checker

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

from pandora2d.common import dataset_disp_maps
from . import refinement, fo_cython

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@refinement.AbstractRefinement.register_subclass("optical_flow")
class OpticalFlow(refinement.AbstractRefinement):
    """
    OpticalFLow class allows to perform the subpixel cost refinement step
    """

    _WINDOW_SIZE = None
    _NBR_ITERATION = 1

    def __init__(self, **cfg: str) -> None:
        """
        :param img_left: xr.Dataset containing :
        - im : 2D (row, col) xr.DataArray
        - msk : 2D (row, col) xr.DataArray
        :type img_left: xr.Dataset
        :param img_right: xr.Dataset containing :
        - im : 2D (row, col) xr.DataArray
        - msk : 2D (row, col) xr.DataArray
        :type img_right: xr.Dataset
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg["window_size"]
        self._nbr_iteration = self.cfg["nbr_iteration"]

    def check_conf(self, **cfg: str) -> Dict[str, str]:
        """
        Check the refinement configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE

        if "nbr_iteration" not in cfg:
            cfg["nbr_iteration"] = self._NBR_ITERATION  # type: ignore

        schema = {
            "refinement_method": And(str, lambda x: x in ["optical_flow"]),
            "window_size": Or(None, And(int, lambda ws: ws > 0, lambda ws: ws % 2 != 0)),
            "nbr_iteration": And(int, lambda nbr_i: nbr_i > 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    @staticmethod
    def optical_flow(
        left: np.ndarray, right: np.ndarray, window_size: int, pixel_maps: xr.Dataset, disp_range: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lucas & Kanade's optical flow algorithm
        :param left: left image
        :type left: np.array
        :param right: right image
        :type right: np.array
        :param window_size: size of window
        :type window_size: int
        :param pixel_maps: dataset of pixel disparity maps
        :type pixel_maps: xr.Dataset
        :param disp_range: bounds of disp range decided by user
        :type disp_range: list
        :return: delta_col, delta_row: subpixel disparity maps
        :rtype: Tuple[np.array, np.array]
        """

        dec_row = pixel_maps["row_map"].data
        dec_col = pixel_maps["col_map"].data
        invalid_disparity = pixel_maps.attrs["invalid_disparity"]

        # gradients measure
        grad_x = np.gradient(left, axis=1)
        grad_y = np.gradient(left, axis=0)
        grad_t = left - right

        # marge measure
        w = int(window_size / 2)

        dec_row, dec_col = fo_cython.optical_flow(
            grad_x, grad_y, grad_t, w, left, invalid_disparity, dec_row, dec_col
        )

        # replace values outside user's range
        dec_col[dec_col < disp_range[0]] = invalid_disparity
        dec_col[dec_col > disp_range[1]] = invalid_disparity
        dec_row[dec_row < disp_range[2]] = invalid_disparity
        dec_row[dec_row > disp_range[3]] = invalid_disparity

        return dec_row, dec_col

    @staticmethod
    def warped_img(img: xr.Dataset, pixel_maps: xr.Dataset) -> np.ndarray:
        """
        Warp image pixel by pixel
        :param img: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
        - msk : 2D (row, col) xarray.DataArray
        :type img: xarray.dataset
        :param pixel_maps: dataset of pixel disparity maps
        :type pixel_maps: xr.Dataset
        :return: new_img: warped image
        :rtype: np.array
        """
        nbr_row, nbr_col = img["im"].data.shape
        row_map = pixel_maps["row_map"].data
        col_map = pixel_maps["col_map"].data

        x, y = np.meshgrid(range(nbr_col), range(nbr_row))

        new_img = map_coordinates(img["im"].data, [y + row_map, x + col_map], cval=-10000)
        return new_img

    def refinement_method(
        self,
        cost_volumes: xr.Dataset,
        pixel_maps: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None
    ) -> Tuple[np.array, np.array]:
        """
        Compute refine disparity maps
        :param cost_volumes: Cost_volumes has (row, col, disp_col, disp_row) dimensions
        :type cost_volumes: xr.Dataset
        :param pixel_maps: dataset of pixel disparity maps
        :type pixel_maps: xr.Dataset
        :param img_left: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
        - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.dataset
        :param img_right: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
        - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.dataset
        :return: delta_col, delta_row: subpixel disparity maps
        :rtype: Tuple[np.array, np.array]
        """

        disp_range = [
            cost_volumes["disp_col"].data[0],
            cost_volumes["disp_col"].data[-1],
            cost_volumes["disp_row"].data[0],
            cost_volumes["disp_row"].data[-1],
        ]

        # first warp after pixel research
        warped_right = self.warped_img(img_right, pixel_maps)

        if self._window_size is None:
            self._window_size = cost_volumes.attrs["window_size"]

        # first warp after pixel research
        warped_right = self.warped_img(img_right, pixel_maps)
        invalid_disparity = pixel_maps.attrs["invalid_disparity"]

        for _ in range(self._nbr_iteration):  # type: ignore
            delta_row, delta_col = self.optical_flow(
                img_left["im"].data, warped_right, cost_volumes.attrs["window_size"], pixel_maps, disp_range
            )
            pixel_maps = dataset_disp_maps(delta_row, delta_col)
            pixel_maps.attrs["invalid_disparity"] = invalid_disparity

            warped_right = self.warped_img(img_right, pixel_maps)

        return delta_col, delta_row
