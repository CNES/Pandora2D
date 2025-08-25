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
with mutual information  and zncc methods.
"""

from json_checker import And

import numpy as np
import xarray as xr

from pandora2d.margins import Margins

from pandora2d.img_tools import shift_subpix_img_2d
from pandora2d.margins import UniformMargins
from pandora2d.matching_cost.registry import MatchingCostRegistry
from .base import BaseMatchingCost

from ..common_cpp import common_bind
from ..matching_cost_cpp import matching_cost_bind


@MatchingCostRegistry.add("mutual_information")
@MatchingCostRegistry.add("zncc")
class CorrelationMethods(BaseMatchingCost):
    """
    Mutual Information class
    """

    @property
    def margins(self) -> Margins:
        """Return matching costs' Margins."""
        return UniformMargins(int((self._window_size - 1) / 2))

    @property
    def schema(self):
        schema = super().schema

        schema.update(
            {
                "matching_cost_method": And(str, lambda x: x in ["zncc", "mutual_information"]),
                "float_precision": And(str, lambda x: np.dtype(x) in [np.float32, np.float64]),
            }
        )

        return schema

    def set_shifted_right_images(self, img_right: xr.Dataset) -> None:
        """
        Compute shifted by subpix right image and assign `shifted_right_images` attribute.

        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :return: None
        """
        self.shifted_right_images = shift_subpix_img_2d(img_right, self._subpix)

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

        # Add type measure to attributes for WTA
        self.cost_volumes.attrs["type_measure"] = "max"

        imgs_right = [right["im"].values for right in self.shifted_right_images]
        offset_cv_img_row = self.cost_volumes.row.data[0] - img_left.row.data[0]
        offset_cv_img_col = self.cost_volumes.col.data[0] - img_left.col.data[0]

        if np.issubdtype(self.cost_volumes["cost_volumes"].data.dtype, np.float32):
            compute_cost_volumes_cpp = matching_cost_bind.compute_cost_volumes_cpp_float
        elif np.issubdtype(self.cost_volumes["cost_volumes"].data.dtype, np.float64):
            compute_cost_volumes_cpp = matching_cost_bind.compute_cost_volumes_cpp_double
        else:
            raise TypeError("Cost volume must be in np.float32 or np.float64")

        # Call compute_cost_volumes_cpp
        compute_cost_volumes_cpp(
            img_left["im"].data,
            imgs_right,
            self.cost_volumes["cost_volumes"].data,
            self.cost_volumes["criteria"].data,
            common_bind.CostVolumeSize(*self.cost_volumes["cost_volumes"].shape),
            self.cost_volumes.disp_row.data,
            self.cost_volumes.disp_col.data,
            offset_cv_img_row,
            offset_cv_img_col,
            self.cost_volumes.attrs["window_size"],
            self.cost_volumes.attrs["step"],
            self.cost_volumes.attrs["measure"],
        )

        self.set_out_of_disparity_range_to_other_value(img_left, -np.inf)

        return self.cost_volumes
