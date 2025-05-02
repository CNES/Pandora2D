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
This module contains functions associated to the matching cost computation step.
"""

import copy
from typing import Dict, List, Union

import numpy as np
import xarray as xr
from pandora import matching_cost
from pandora.margins import Margins

from pandora2d import img_tools
from pandora2d.common import (
    set_out_of_col_disparity_range_to_other_value,
    set_out_of_row_disparity_range_to_other_value,
)

from .base import BaseMatchingCost


class PandoraMatchingCostMethods(BaseMatchingCost):
    """
    Matching Cost class
    """

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        super().__init__(cfg)
        # Check the matching_cost parameters specific to pandora
        self.pandora_matching_cost_ = matching_cost.AbstractMatchingCost(  # type: ignore[abstract]
            **self.get_config_for_pandora(cfg)
        )

        self.grid: Union[xr.Dataset, None] = None

    @property
    def cfg(self) -> Dict[str, Union[str, int, List[int]]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        :rtype: cfg: dict
        """
        return {
            "matching_cost_method": self._method,
            "step": self.step,
            "window_size": self.window_size,
            "subpix": self._subpix,
            "spline_order": self._spline_order,
        }

    @property
    def window_size(self) -> int:
        """
        Get window_size, parameter specific to pandora

        :return: window_size: window used to compute correlation
        :rtype: window_size: int
        """
        return self.pandora_matching_cost_._window_size  # pylint: disable=W0212 protected-access

    @window_size.setter
    def window_size(self, value) -> None:
        """
        Set window_size, parameter specific to pandora

        :return: window_size: window used to compute correlation
        :rtype: window_size: int
        """
        # Does nothing as we just want to override superclass behavior

    @property
    def _subpix(self) -> int:
        """
        Get subpix, parameter specific to pandora

        :return: subpix: subpix used
        :rtype: subpix: int
        """
        return self.pandora_matching_cost_._subpix  # pylint: disable=W0212 protected-access

    @_subpix.setter
    def _subpix(self, value: int) -> int:
        """
        Set subpix, parameter specific to pandora

        :return: subpix: subpix used
        :rtype: subpix: int
        """
        # Does nothing as we just want to override superclass behavior

    @property
    def _spline_order(self) -> int:
        """
        Get spline_order, parameter specific to pandora

        :return: spline_order: spline_order used
        :rtype: spline_order: int
        """
        return self.pandora_matching_cost_._spline_order  # pylint: disable=W0212 protected-access

    @_spline_order.setter
    def _spline_order(self, value: int) -> int:
        """
        Set spline_order, parameter specific to pandora

        :return: spline_order: spline_order used
        :rtype: spline_order: int
        """
        # Does nothing as we just want to override superclass behavior

    @property
    def margins(self) -> Margins:
        """
        Get margins from pandora correlation measurement

        """
        return self.pandora_matching_cost_.margins

    def get_config_for_pandora(self, cfg: Dict) -> Dict[str, str]:
        """
        Get configuration for matching_cost in pandora

        :param cfg: user_config for matching cost pandora2d
        :type cfg: dict
        :return: cfg: matching cost pandora configuration
        :rtype: cfg: dict
        """
        copy_cfg = copy.deepcopy(cfg)
        copy_cfg["step"] = self._step_col
        return copy_cfg

    def allocate(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cfg: Dict,
        margins: Margins = None,
    ) -> None:
        """

        Allocate the cost volume for pandora

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :param margins: refinement margins
        :type margins: Margins
        :return: None
        """

        grid_min_col = img_left["col_disparity"].sel(band_disp="min").data.copy()
        grid_max_col = img_left["col_disparity"].sel(band_disp="max").data.copy()

        if margins is not None:
            grid_min_col -= margins.left
            grid_max_col += margins.right

        # Get updated ROI left margin for pandora method get_coordinates()
        # To get right coordinates in cost_volume when initial left_margin > cfg["ROI"]["col"]["first"]
        # We need to have left_margin = cfg["ROI"]["col"]["first"]
        cfg_for_get_coordinates = BaseMatchingCost.cfg_for_get_coordinates(cfg)

        # Initialize pandora an empty grid for cost volume
        self.grid = self.pandora_matching_cost_.allocate_cost_volume(
            img_left, (grid_min_col, grid_max_col), cfg_for_get_coordinates
        )

        super().allocate(img_left, img_right, cfg, margins)

    def compute_cost_volumes(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        margins: Margins = None,
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
        :param margins: refinement margins
        :type margins: Margins
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        grid_min_col = img_left["col_disparity"].sel(band_disp="min").data.copy()
        grid_max_col = img_left["col_disparity"].sel(band_disp="max").data.copy()

        if margins is not None:
            grid_min_col -= margins.left
            grid_max_col += margins.right

        # Get disparity coordinates for cost_volumes
        disps_row = self.get_disp_row_coords(img_left, margins)

        row_index = self.cost_volumes.coords["row"] - img_left.coords["row"].data[0]

        # Contains the shifted right images (with subpixel)
        imgs_right_shift_subpixel = img_tools.shift_subpix_img(img_right, self._subpix, order=self._spline_order)

        for idx, disp_row in enumerate(disps_row):
            i_right = int((disp_row % 1) * self._subpix)

            # Images contained in imgs_right_shift_subpixel are already shifted by 1/subpix.
            # In order for img_right_shift to contain the right image shifted from disp_row,
            # we call img_tools.shift_disp_row_img with np.floor(disp_row).

            # For example if subpix=2 and disp_row=1.5
            # i_right=1
            # imgs_right_shift_subpixel[i_right] is shifted by 0.5
            # In img_tools.shift_disp_row_img we shift it by np.floor(1.5)=1 --> In addition it is shifted by 1.5

            # Another example if subpix=4 and disp_row=-1.25
            # i_right=3
            # imgs_right_shift_subpixel[i_right] is shifted by 0.75
            # In img_tools.shift_disp_row_img we shift it by np.floor(-1.25)=-2 --> In addition it is shifted by -1.25

            # Shift image in the y axis
            img_right_shift = img_tools.shift_disp_row_img(imgs_right_shift_subpixel[i_right], np.floor(disp_row))

            # Compute cost volume
            cost_volume = self.pandora_matching_cost_.compute_cost_volume(img_left, img_right_shift, self.grid)

            # Add current cost volume to the cost_volumes dataset
            self.cost_volumes["cost_volumes"].data[:, :, idx, :] = cost_volume["cost_volume"].data[row_index, :, :]

        # Add type measure to attributes for WTA
        self.cost_volumes.attrs["type_measure"] = cost_volume.attrs["type_measure"]

        # Select correct rows and columns in case of a step different from 1.
        row_cv = self.cost_volumes.row.values
        col_cv = self.cost_volumes.col.values

        set_out_of_row_disparity_range_to_other_value(
            self.cost_volumes["cost_volumes"],
            img_left["row_disparity"].sel(band_disp="min", row=row_cv, col=col_cv).data,
            img_left["row_disparity"].sel(band_disp="max", row=row_cv, col=col_cv).data,
            np.nan,
            self.cost_volumes.attrs["row_disparity_source"],
        )
        set_out_of_col_disparity_range_to_other_value(
            self.cost_volumes["cost_volumes"],
            img_left["col_disparity"].sel(band_disp="min", row=row_cv, col=col_cv).data,
            img_left["col_disparity"].sel(band_disp="max", row=row_cv, col=col_cv).data,
            np.nan,
            self.cost_volumes.attrs["col_disparity_source"],
        )

        return self.cost_volumes
