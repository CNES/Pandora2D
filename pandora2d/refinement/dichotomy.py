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
"""
Module for Dichotomy refinement method.
"""
import logging
from typing import Dict

import numpy as np
import xarray as xr

from json_checker import And

from pandora.margins import Margins
from . import refinement


@refinement.AbstractRefinement.register_subclass("dichotomy")
class Dichotomy(refinement.AbstractRefinement):
    """Subpixel refinement method by dichotomy."""

    NB_MAX_ITER = 9
    schema = {
        "refinement_method": And(str, lambda x: x in ["dichotomy"]),
        "iterations": And(int, lambda it: it > 0),
        "filter": And(str, lambda x: x in ["sinc", "bicubic", "spline"]),
    }

    @classmethod
    def check_conf(cls, cfg: Dict) -> Dict:
        """
        Check the refinement method configuration.

        Will change `number_of_iterations` value by `Dichotomy.NB_MAX_ITER` if above `Dichotomy.NB_MAX_ITER`.

        :param cfg: user_config for refinement method
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        cfg = super().check_conf(cfg)
        if cfg["iterations"] > cls.NB_MAX_ITER:
            logging.warning(
                "number_of_iterations %s is above maximum iteration. Maximum value of %s will be used instead.",
                cfg["iterations"],
                cls.NB_MAX_ITER,
            )
            cfg["iterations"] = cls.NB_MAX_ITER
        return cfg

    @property
    def margins(self):
        """
        Create margins for dichotomy object
        """
        return Margins(2, 2, 2, 2)

    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> None:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixel disparity maps
        :type disp_map: xarray.Dataset
        :param img_left: left image dataset
        :type img_left: xarray.Dataset
        :param img_right: right image dataset
        :type img_right: xarray.Dataset
        :return: the refined disparity maps
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        logging.warning("refinement_method of Dichotomy not yet implemented")


class DichotomyWindows:
    """
    Container to extract subsampling cost surfaces around a given disparity from cost volumes.

    Dichotomy Window of point with coordinates `row==0` and `col==1` can be accessed with `dichotomy_window[0, 1]`.

    The container is iterable row first then columns.
    """

    def __init__(self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, disparity_margins: Margins):
        """
        Extract subsampling cost surfaces from cost volumes around a given disparity from cost volumes.

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixels disparity maps
        :param disparity_margins: margins used to define disparity ranges
        :type disparity_margins: Margins
        """
        self.cost_volumes = cost_volumes
        self.min_row_disp_map = disp_map["row_map"] - disparity_margins.up
        self.max_row_disp_map = disp_map["row_map"] + disparity_margins.down
        self.min_col_disp_map = disp_map["col_map"] - disparity_margins.left
        self.max_col_disp_map = disp_map["col_map"] + disparity_margins.right

    def __getitem__(self, item):
        """Get cost surface of coordinates item where item is (row, col)."""
        row, col = item
        row_slice = np.s_[self.min_row_disp_map.sel(row=row, col=col) : self.max_row_disp_map.sel(row=row, col=col)]
        col_slice = np.s_[self.min_col_disp_map.sel(row=row, col=col) : self.max_col_disp_map.sel(row=row, col=col)]
        return self.cost_volumes["cost_volumes"].sel(row=row, col=col, disp_row=row_slice, disp_col=col_slice)

    def __iter__(self):
        """Iter over cost surfaces, row first then columns."""
        for row in self.cost_volumes.coords["row"]:
            for col in self.cost_volumes.coords["col"]:
                yield self[row, col]
