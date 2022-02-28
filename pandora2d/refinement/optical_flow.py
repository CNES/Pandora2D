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
This module contains functions associated to the optical flow method used in the refinement step.
"""

from typing import Dict, Tuple
import itertools
from json_checker import And, Checker

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

from . import refinement


@refinement.AbstractRefinement.register_subclass("optical_flow")
class OpticalFlow(refinement.AbstractRefinement):
    """
    OpticalFLow class allows to perform the subpixel cost refinement step
    """

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

    @staticmethod
    def check_conf(**cfg: str) -> Dict[str, str]:
        """
        Check the refinement configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        schema = {
            "refinement_method": And(str, lambda x: x in ["optical_flow"]),
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

        # gradients measure
        grad_x = np.gradient(left, axis=1)
        grad_y = np.gradient(left, axis=0)
        grad_t = left - right

        # marge measure
        w = int(window_size / 2)

        for i, j in itertools.product(range(w, left.shape[0] - w), range(w, left.shape[1] - w)):

            # Select pixel and neighbourhoods
            Ix = grad_x[i - w : i + w + 1, j - w : j + w + 1]
            Iy = grad_y[i - w : i + w + 1, j - w : j + w + 1]
            It = grad_t[i - w : i + w + 1, j - w : j + w + 1]

            # Create A et B matrix for Lucas Kanade
            A = np.vstack((Ix.flatten(), Iy.flatten())).T
            B = np.reshape(It, len(It) ** 2)[np.newaxis].T

            # v = (A^T.A)^-1.A^T.B
            try:
                motion = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B))
            # if matrix is full of NaN or 0
            except np.linalg.LinAlgError:
                motion = [np.nan, np.nan]

            dec_row[i, j] = dec_row[i, j] + motion[1]
            dec_col[i, j] = dec_col[i, j] + motion[0]

        # erase nonsense values
        dec_row = np.clip(dec_row, disp_range[0], disp_range[1])
        dec_col = np.clip(dec_col, disp_range[2], disp_range[3])

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
            cost_volumes["disp_row"].data[0],
            cost_volumes["disp_row"].data[-1],
            cost_volumes["disp_col"].data[0],
            cost_volumes["disp_col"].data[-1],
        ]

        warped_right = self.warped_img(img_right, pixel_maps)
        delta_row, delta_col = self.optical_flow(
            img_left["im"].data, warped_right, cost_volumes.attrs["window_size"], pixel_maps, disp_range
        )

        return delta_col, delta_row
