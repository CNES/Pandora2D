#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
from pandora.margins import Margins


from pandora2d import img_tools
import pandora2d.schema as cst_schema
from pandora2d.common import (
    set_out_of_row_disparity_range_to_other_value,
    set_out_of_col_disparity_range_to_other_value,
)


class MatchingCost:
    """
    Matching Cost class
    """

    _STEP = [1, 1]

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        # Check the matching_cost parameters specific to pandora2d
        updated_cfg = self.check_conf(cfg)
        self._matching_cost_method = updated_cfg["matching_cost_method"]
        # Cast to int in order to help mypy because self.cfg is a Dict, and it can not know the type of step.
        self._step_row = cast(int, updated_cfg["step"][0])
        self._step_col = cast(int, updated_cfg["step"][1])

        # Check the matching_cost parameters specific to pandora
        self.pandora_matching_cost_ = matching_cost.AbstractMatchingCost(**self.get_config_for_pandora(cfg))
        self.grid_: xr.Dataset = None

    @property
    def cfg(self) -> Dict[str, Union[str, int, List[int]]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        :rtype: cfg: dict
        """
        return {
            "matching_cost_method": self._matching_cost_method,
            "step": self._step,
            "window_size": self._window_size,
            "subpix": self._subpix,
            "spline_order": self._spline_order,
        }

    @property
    def _step(self) -> List[int]:
        """
        Get step [row, col]

        :return: step: list with row & col step
        :rtype: step: list
        """
        return [self._step_row, self._step_col]

    @property
    def _window_size(self) -> int:
        """
        Get window_size, parameter specific to pandora

        :return: window_size: window used to compute correlation
        :rtype: window_size: int
        """
        return self.pandora_matching_cost_._window_size  # pylint: disable=W0212 protected-access

    @property
    def _subpix(self) -> int:
        """
        Get subpix, parameter specific to pandora

        :return: subpix: subpix used
        :rtype: subpix: int
        """
        return self.pandora_matching_cost_._subpix  # pylint: disable=W0212 protected-access

    @property
    def _spline_order(self) -> int:
        """
        Get spline_order, parameter specific to pandora

        :return: spline_order: spline_order used
        :rtype: spline_order: int
        """
        return self.pandora_matching_cost_._spline_order  # pylint: disable=W0212 protected-access

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

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """
        Check the matching cost configuration

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        # Check only matching_cost pandora2d param
        pandora_2d_cfg = {
            "matching_cost_method": cfg["matching_cost_method"],
            "step": cfg.get("step", self._STEP),
        }
        schema = {
            "matching_cost_method": And(str, lambda mc: mc in ["ssd", "sad", "zncc", "mc_cnn"]),
            "step": cst_schema.STEP_SCHEMA,
        }

        checker = Checker(schema)
        checker.validate(pandora_2d_cfg)

        return pandora_2d_cfg

    @staticmethod
    def allocate_cost_volumes(
        cost_volume_attr: dict,
        row: np.ndarray,
        col: np.ndarray,
        disp_range_col: np.ndarray,
        disp_range_row: np.ndarray,
        np_data: np.ndarray = None,
    ) -> xr.Dataset:
        """
        Allocate the cost volumes

        :param cost_volume_attr: the cost_volume's attributes product by Pandora
        :type cost_volume_attr: xr.Dataset
        :param row: dimension of the image (row)
        :type row: np.ndarray
        :param col: dimension of the image (columns)
        :type col: np.ndarray
        :param disp_range_col: columns disparity range.
        :type disp_range_col: np.ndarray
        :param disp_range_row: rows disparity range.
        :type disp_range_row: np.ndarray
        :param np_data: 4D numpy.ndarray og cost_volumes. Defaults to None.
        :type np_data: np.ndarray
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        # Create the cost volume
        if np_data is None:
            np_data = np.zeros((len(row), len(col), len(disp_range_col), len(disp_range_row)), dtype=np.float32)

        cost_volumes = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], np_data)},
            coords={"row": row, "col": col, "disp_col": disp_range_col, "disp_row": disp_range_row},
        )

        cost_volumes.attrs = cost_volume_attr

        # del pandora attributes
        del cost_volumes.attrs["col_to_compute"]
        del cost_volumes.attrs["sampling_interval"]

        return cost_volumes

    def allocate_cost_volume_pandora(
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
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :param margins: refinement margins
        :type margins: Margins
        :return: None
        """
        # Adapt Pandora matching cost configuration
        img_left.attrs["disparity_source"] = img_left.attrs["col_disparity_source"]
        grid_min_col = img_left["col_disparity"].sel(band_disp="min").data.copy()
        grid_max_col = img_left["col_disparity"].sel(band_disp="max").data.copy()

        if margins is not None:
            grid_min_col -= margins.left
            grid_max_col += margins.right

        # Get updated ROI left margin for pandora method get_coordinates()
        # To get right coordinates in cost_volume when initial left_margin > cfg["ROI"]["col"]["first"]
        # We need to have left_margin = cfg["ROI"]["col"]["first"]
        cfg_for_get_coordinates = copy.deepcopy(cfg)
        if "ROI" in cfg:
            cfg_for_get_coordinates["ROI"]["margins"] = (
                min(cfg["ROI"]["margins"][0], cfg["ROI"]["col"]["first"]),
                cfg["ROI"]["margins"][1],
                cfg["ROI"]["margins"][2],
                cfg["ROI"]["margins"][3],
            )

        # Initialize pandora an empty grid for cost volume
        self.grid_ = self.pandora_matching_cost_.allocate_cost_volume(
            img_left, (grid_min_col, grid_max_col), cfg_for_get_coordinates
        )

        # Compute validity mask to identify invalid points in cost volume
        self.grid_ = validity_mask(img_left, img_right, self.grid_)

        # Add ROI margins in attributes
        # Enables to compute cost volumes row coordinates later by using pandora.matching_cost.get_coordinates()
        # Get updated ROI up margin for pandora method get_coordinates()
        # To get right coordinates in cost_volume when initial up_margin > cfg["ROI"]["row"]["first"]
        # We need to have up_margin = cfg["ROI"]["row"]["first"]
        if "ROI" in cfg:
            self.grid_.attrs["ROI_margins_for_cv"] = (
                cfg["ROI"]["margins"][0],
                min(cfg["ROI"]["margins"][1], cfg["ROI"]["row"]["first"]),
                cfg["ROI"]["margins"][2],
                cfg["ROI"]["margins"][3],
            )
        else:
            self.grid_.attrs["ROI_margins_for_cv"] = None

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
        :param grid_min_col: grid containing min disparities for columns.
        :type grid_min_col: np.ndarray
        :param grid_max_col: grid containing max disparities for columns.
        :type grid_max_col: np.ndarray
        :param grid_min_row: grid containing min disparities for rows.
        :type grid_min_row: np.ndarray
        :param grid_max_row: grid containing max disparities for rows.
        :type grid_max_row: np.ndarray
        :param margins: refinement margins
        :type margins: Margins
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        cost_volumes = xr.Dataset()

        # Adapt Pandora matching cost configuration
        img_left.attrs["disparity_source"] = img_left.attrs["col_disparity_source"]
        grid_min_col = img_left["col_disparity"].sel(band_disp="min").data.copy()
        grid_max_col = img_left["col_disparity"].sel(band_disp="max").data.copy()
        grid_min_row = img_left["row_disparity"].sel(band_disp="min").data.copy()
        grid_max_row = img_left["row_disparity"].sel(band_disp="max").data.copy()

        if margins is not None:
            grid_min_col -= margins.left
            grid_max_col += margins.right
            grid_min_row -= margins.up
            grid_max_row += margins.down

        # Obtain absolute min and max disparities
        min_row, max_row = self.pandora_matching_cost_.get_min_max_from_grid(grid_min_row, grid_max_row)
        min_col, max_col = self.pandora_matching_cost_.get_min_max_from_grid(grid_min_col, grid_max_col)

        # Array with all x disparities
        disps_col = self.pandora_matching_cost_.get_disparity_range(min_col, max_col, self._subpix)
        # Array with all y disparities
        disps_row = self.pandora_matching_cost_.get_disparity_range(min_row, max_row, self._subpix)

        row_index = None

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
            cost_volume = self.pandora_matching_cost_.compute_cost_volume(img_left, img_right_shift, self.grid_)
            # Mask cost volume
            self.pandora_matching_cost_.cv_masked(img_left, img_right_shift, cost_volume, grid_min_col, grid_max_col)
            # If first iteration, initialize cost_volumes dataset
            if idx == 0:
                img_row_coordinates = img_left["im"].coords["row"].values

                # Case without a ROI: we only take the step into account to compute row coordinates.
                if self.grid_.attrs["ROI_margins_for_cv"] is None:
                    row_coords = np.arange(img_row_coordinates[0], img_row_coordinates[-1] + 1, self._step_row)

                # Case with a ROI: we use pandora get_coordinates() method to compute row coordinates.
                # This method consider step and ROI margins when computing row coordinates.
                # This ensures that the first point of the ROI given by the user is computed in the cost volume.
                else:
                    row_coords = self.pandora_matching_cost_.get_coordinates(
                        margin=self.grid_.attrs["ROI_margins_for_cv"][1],
                        img_coordinates=img_row_coordinates,
                        step=self._step_row,
                    )

                # We want row_index to start at 0
                row_index = row_coords - img_left.coords["row"].data[0]

                # Columns coordinates are already handled correctly by Pandora.
                col_coords = cost_volume["cost_volume"].coords["col"]

                cost_volumes = self.allocate_cost_volumes(
                    cost_volume.attrs, row_coords, col_coords, disps_col, disps_row, None
                )

            # Add current cost volume to the cost_volumes dataset
            cost_volumes["cost_volumes"].data[:, :, :, idx] = cost_volume["cost_volume"].data[row_index, :, :]

        # Add disparity source
        del cost_volumes.attrs["disparity_source"]
        cost_volumes.attrs["col_disparity_source"] = img_left.attrs["col_disparity_source"]
        cost_volumes.attrs["row_disparity_source"] = img_left.attrs["row_disparity_source"]
        cost_volumes.attrs["disparity_margins"] = margins
        cost_volumes.attrs["step"] = self._step

        # Delete ROI_margins attributes which we used to calculate the row coordinates in the cost_volumes
        del cost_volumes.attrs["ROI_margins_for_cv"]

        set_out_of_row_disparity_range_to_other_value(
            cost_volumes["cost_volumes"],
            img_left["row_disparity"].sel(band_disp="min").data,
            img_left["row_disparity"].sel(band_disp="max").data,
            np.nan,
            cost_volumes.attrs["row_disparity_source"],
        )
        set_out_of_col_disparity_range_to_other_value(
            cost_volumes["cost_volumes"],
            img_left["col_disparity"].sel(band_disp="min").data,
            img_left["col_disparity"].sel(band_disp="max").data,
            np.nan,
            cost_volumes.attrs["col_disparity_source"],
        )

        return cost_volumes
