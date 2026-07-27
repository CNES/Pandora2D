# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

import logging

import numpy as np
import xarray as xr
from json_checker import And

from pandora2d.img_tools import shift_subpix_img_2d
from pandora2d.margins import Margins, UniformMargins
from pandora2d.matching_cost.registry import MatchingCostRegistry
from pandora2d.common import get_disparity_grids

from ..common_cpp import common_bind
from ..matching_cost_cpp import matching_cost_bind
from .base import BaseMatchingCost


def select_zncc_optim_method(window_size: int, step: list[int]) -> str:
    """
    Select the most appropriate ZNCC C++ implementation from window size and step.

    The ``window_size / max(step) > 3`` threshold was determined empirically by measuring the execution
    time of "zncc-optim-1" and "zncc-optim-2" on various window_size/step configurations
    (see :ref:`matching_cost` in the user guide).

    :param window_size: correlation window size
    :param step: step [row, col] for cost volume computation
    :return: "zncc-optim-1" or "zncc-optim-2"
    """
    max_step = max(step)
    if window_size / max_step > 3:
        return "zncc-optim-1"
    return "zncc-optim-2"


@MatchingCostRegistry.add("mutual_information")
@MatchingCostRegistry.add("zncc")
@MatchingCostRegistry.add("zncc-optim-1")
@MatchingCostRegistry.add("zncc-optim-2")
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
                "matching_cost_method": And(
                    str, lambda x: x in ["zncc", "zncc-optim-1", "zncc-optim-2", "mutual_information"]
                ),
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
        :return: None
        """
        self.shifted_right_images = shift_subpix_img_2d(img_right, self._subpix, order=self._spline_order)

    def _resolve_cpp_correlation_method(self) -> str:
        """
        Resolve the C++ correlation method to pass to compute_cost_volumes_cpp.

        :return: C++ correlation method name
        """
        if self._method in ("mutual_information", "zncc-optim-1", "zncc-optim-2"):
            return self._method
        if self._method == "zncc":
            selected_method = select_zncc_optim_method(self._window_size, self.step)
            logging.info("Auto-selected ZNCC implementation: %s", selected_method)
            return selected_method
        raise ValueError(f"Unsupported correlation method: {self._method}")

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
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param margins: refinement margins
        :return: cost_volumes: 4D Dataset containing the cost_volumes
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

        cv_coords = (self.cost_volumes.row.values, self.cost_volumes.col.values)

        min_disp_row, max_disp_row, min_disp_col, max_disp_col = get_disparity_grids(img_left, cv_coords)

        if margins is not None:
            min_disp_row -= margins.up
            max_disp_row += margins.down
            min_disp_col -= margins.left
            max_disp_col += margins.right

        cpp_correlation_method = self._resolve_cpp_correlation_method()

        # Call compute_cost_volumes_cpp
        compute_cost_volumes_cpp(
            img_left["im"].data,
            min_disp_row,
            max_disp_row,
            min_disp_col,
            max_disp_col,
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
            cpp_correlation_method,
        )

        self.set_out_of_disparity_range_to_other_value(img_left, -np.inf)

        return self.cost_volumes
