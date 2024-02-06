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
Module for Dichotomy refinement method.
"""
import logging
from typing import Dict

import xarray as xr

from json_checker import And
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

    def refinement_method(self, cost_volumes: xr.Dataset, pixel_maps: xr.Dataset) -> None:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.dataset
        :param pixel_maps: pixels disparity maps
        :type pixel_maps: xarray.dataset
        :return: the refined disparity maps
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        logging.warning("refinement_method of Dichotomy not yet implemented")
