# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the matching cost computation step
with mutual information method.
"""

from typing import Dict, Union, Tuple
from json_checker import And

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pandora import matching_cost as pandora_matching_cost
from pandora.margins import Margins
from pandora.margins.descriptors import HalfWindowMargins

from pandora2d.img_tools import shift_subpix_img_2d
from pandora2d.criteria import get_criteria_dataarray
from pandora2d.matching_cost.registry import MatchingCostRegistry
from .base import BaseMatchingCost

from ..common_cpp import common_bind
from ..matching_cost_cpp import matching_cost_bind


@MatchingCostRegistry.add("mutual_information")
class MutualInformation(BaseMatchingCost):
    """
    Mutual Information class
    """

    margins = HalfWindowMargins()

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of mutual information class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """

        super().__init__(cfg)

        self._method = str(self.cfg["matching_cost_method"])
        self.grid_4d: Union[xr.Dataset, None] = None

    @property
    def schema(self):
        schema = super().schema

        schema.update(
            {
                "matching_cost_method": And(str, lambda x: x in ["mutual_information"]),
            }
        )

        return schema

    def get_cv_row_col_coords(
        self, img_row_coordinates: NDArray, img_col_coordinates: NDArray, cfg: Dict
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute cost_volumes row and col coordinates according to image coordinates

        :param img_row_coordinates: row coordinates of left image
        :type img_row_coordinates: NDArray
        :param img_col_coordinates: col coordinates of left image
        :type img_col_coordinates: NDArray
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :return: a Tuple of np.ndarray that contains the right coordinates for row and col
        :rtype: Tuple[NDArray, NDArray]
        """

        # Get updated ROI left/up margin for get_coordinates() method
        # To get right coordinates in cost_volume when initial left_margin > cfg["ROI"]["col"]["first"]
        # or initial up_margin > cfg["ROI"]["row"]["first"]
        # We need to have left_margin = cfg["ROI"]["col"]["first"] and up_margin = cfg["ROI"]["row"]["first"]
        cfg_for_get_coordinates = BaseMatchingCost.cfg_for_get_coordinates(cfg)

        # Get correct coordinates to be sure to process the first point of ROI
        if "ROI" in cfg:
            col_coords = pandora_matching_cost.AbstractMatchingCost.get_coordinates(
                margin=cfg_for_get_coordinates["ROI"]["margins"][0],
                img_coordinates=img_col_coordinates,
                step=self._step_col,
            )

            row_coords = pandora_matching_cost.AbstractMatchingCost.get_coordinates(
                margin=cfg_for_get_coordinates["ROI"]["margins"][1],
                img_coordinates=img_row_coordinates,
                step=self._step_row,
            )
        else:
            row_coords = np.arange(img_row_coordinates[0], img_row_coordinates[-1] + 1, self._step_row)
            col_coords = np.arange(img_col_coordinates[0], img_col_coordinates[-1] + 1, self._step_col)

        return row_coords, col_coords

    def allocate(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cfg: Dict,
        margins: Margins = None,
    ) -> None:
        """

        Allocate the cost volume

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

        img_row_coordinates = img_left["im"].coords["row"].values
        img_col_coordinates = img_left["im"].coords["col"].values

        row_coords, col_coords = self.get_cv_row_col_coords(img_row_coordinates, img_col_coordinates, cfg)
        # Get disparity coordinates for cost_volumes
        disps_row_coords = self.get_disp_row_coords(img_left, margins)
        disps_col_coords = self.get_disp_col_coords(img_left, margins)

        grid_attrs = img_left.attrs

        grid_attrs.update(
            {
                "window_size": self._window_size,
                "subpixel": self._subpix,
                "offset_row_col": int((self._window_size - 1) / 2),
                "measure": self._method,
                "type_measure": "max",
                "disparity_margins": margins,
                "step": self.step,
            }
        )

        # Allocate 4D cost_volumes
        self.grid_4d = self.allocate_cost_volumes(
            grid_attrs, row_coords, col_coords, disps_row_coords, disps_col_coords, None
        )

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

        self.grid_4d["criteria"] = get_criteria_dataarray(img_left, img_right, self.grid_4d)
        imgs_right_dataset = shift_subpix_img_2d(img_right, self.grid_4d.attrs["subpixel"])

        imgs_right = [right["im"].values for right in imgs_right_dataset]
        cv_values = self.grid_4d["cost_volumes"].data.ravel().astype(np.float64)
        offset_cv_img_row = self.grid_4d.row.data[0] - img_left.row.data[0]
        offset_cv_img_col = self.grid_4d.col.data[0] - img_left.col.data[0]

        # Call compute_cost_volumes_cpp
        matching_cost_bind.compute_cost_volumes_cpp(
            img_left["im"].data,
            imgs_right,
            cv_values,
            common_bind.CostVolumeSize(*self.grid_4d["cost_volumes"].shape),
            self.grid_4d.disp_row.data,
            self.grid_4d.disp_col.data,
            offset_cv_img_row,
            offset_cv_img_col,
            self.grid_4d.attrs["window_size"],
            self.grid_4d.attrs["step"],
            self.grid_4d.attrs["no_data_img"],
        )

        cv_values_reshaped = cv_values.reshape(self.grid_4d["cost_volumes"].shape)
        self.grid_4d["cost_volumes"] = (("row", "col", "disp_row", "disp_col"), cv_values_reshaped)

        return self.grid_4d
