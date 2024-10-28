# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Module for Dichotomy refinement method (cpp version).
"""
import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from json_checker import And

from ..interpolation_filter import AbstractFilter
from . import refinement


@refinement.AbstractRefinement.register_subclass("dichotomy_cpp")
class DichotomyCPP(refinement.AbstractRefinement):
    """Subpixel refinement method by dichotomy (cpp version)."""

    NB_MAX_ITER = 9
    schema = {
        "refinement_method": And(str, lambda x: x in ["dichotomy_cpp"]),
        "iterations": And(int, lambda it: it > 0),
        "filter": And(dict, lambda method: method["method"] in AbstractFilter.interpolation_filter_methods_avail),
    }

    def __init__(self, cfg: dict = None, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        super().__init__(cfg)
        fractional_shift_ = 2 ** -self.cfg["iterations"]
        self.filter = AbstractFilter(  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated
            self.cfg["filter"], fractional_shift=fractional_shift_
        )

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
        Create margins for dichotomy object.

        It will be used for ROI and for dichotomy window extraction from cost volumes.
        """

        return self.filter.margins

    def refinement_method(  # pylint: disable=too-many-locals
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        logging.info(
            "This method is still under development. Currently, only disparity maps and pixel-level cost volume "
            "are being returned."
        )

        col_map = disp_map["col"].data
        row_map = disp_map["row"].data
        cost_values = cost_volumes["cost_volumes"].data

        return col_map, row_map, cost_values
