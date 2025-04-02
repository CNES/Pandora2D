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

from json_checker import And

import numpy as np
import xarray as xr
from pandora.margins import Margins
from pandora.margins.descriptors import HalfWindowMargins

from pandora2d.img_tools import shift_subpix_img_2d
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

    @property
    def schema(self):
        schema = super().schema

        schema.update(
            {
                "matching_cost_method": And(str, lambda x: x in ["mutual_information"]),
            }
        )

        return schema

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
        imgs_right_dataset = shift_subpix_img_2d(img_right, self.cost_volumes.attrs["subpixel"])

        imgs_right = [right["im"].values for right in imgs_right_dataset]
        cv_values = self.cost_volumes["cost_volumes"].data.ravel().astype(np.float64)
        offset_cv_img_row = self.cost_volumes.row.data[0] - img_left.row.data[0]
        offset_cv_img_col = self.cost_volumes.col.data[0] - img_left.col.data[0]

        # Call compute_cost_volumes_cpp
        matching_cost_bind.compute_cost_volumes_cpp(
            img_left["im"].data,
            imgs_right,
            cv_values,
            common_bind.CostVolumeSize(*self.cost_volumes["cost_volumes"].shape),
            self.cost_volumes.disp_row.data,
            self.cost_volumes.disp_col.data,
            offset_cv_img_row,
            offset_cv_img_col,
            self.cost_volumes.attrs["window_size"],
            self.cost_volumes.attrs["step"],
            self.cost_volumes.attrs["no_data_img"],
        )

        cv_values_reshaped = cv_values.reshape(self.cost_volumes["cost_volumes"].shape)
        self.cost_volumes["cost_volumes"] = (("row", "col", "disp_row", "disp_col"), cv_values_reshaped)

        return self.cost_volumes
