#!/usr/bin/env python
#
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
This module contains functions associated to the estimation computation step.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import numpy as np
import xarray as xr


class AbstractEstimation:
    """
    Abstract Estimation class
    """

    __metaclass__ = ABCMeta

    estimation_methods_avail: Dict = {}
    _estimation_method = None
    cfg = None

    def __new__(cls, cfg: dict | None = None):
        """
        Return the plugin associated with the estimation_method given in the configuration

        :param cfg: configuration {'estimation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractEstimation:
            if isinstance(cfg["estimation_method"], str):
                try:
                    return super(AbstractEstimation, cls).__new__(
                        cls.estimation_methods_avail[cfg["estimation_method"]]
                    )
                except KeyError:
                    logging.error("No estimation method named %s supported", cfg["estimation_method"])
                    raise KeyError
            else:
                if isinstance(cfg["estimation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractEstimation, cls).__new__(
                            cls.estimation_methods_avail[cfg["estimation_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No estimation method named %s supported",
                            cfg["estimation_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractEstimation, cls).__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.estimation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    def desc(self) -> None:
        """
        Describes the estimation method
        :return: None
        """
        print(f"{self._estimation_method} estimation measure")

    @abstractmethod
    def compute_estimation(self, img_left: xr.Dataset, img_right: xr.Dataset) -> Tuple[Dict, Dict, np.ndarray, dict]:
        """
        Compute the phase cross correlation method

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :return:row disparity: Dict
                col disparity: Dict
                Calculated shifts: list
                Extra information about estimation : dict
        :rtype: dict, dict, np.ndarray, dict
        """

    @staticmethod
    def update_cfg_with_estimation(
        cfg: Dict, disp_col: Dict, disp_row: Dict, shifts: np.ndarray, extra_dict: dict = None
    ) -> Dict:
        """
        Save calculated shifts in a configuration dictionary

        :param cfg: user configuration
        :type cfg: dict
        :param disp_col: dict with init and range for disparity in column
        :type disp_col: {"init" : int, "range" : int >= 0}
        :param disp_row: dict with init and range for disparity in row
        :type disp_row: {"init" : int, "range" : int >= 0}
        :param shifts: computed global shifts between left and right
        :type shifts: [np.float32, np.float32]
        :param extra_dict: Dictionary containing extra information about estimation
        :type extra_dict: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        cfg["input"]["col_disparity"] = disp_col
        cfg["input"]["row_disparity"] = disp_row

        cfg["pipeline"]["estimation"]["estimated_shifts"] = shifts.tolist()
        if extra_dict is not None:
            for key, value in extra_dict.items():
                cfg["pipeline"]["estimation"][key] = value

        return cfg
