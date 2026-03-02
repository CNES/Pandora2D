#!/usr/bin/env python
#
# Copyright (c) 2026 CS GROUP France
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
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np
import xarray as xr

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class AbstractEstimation:
    """
    Abstract Estimation class
    """

    __metaclass__ = ABCMeta

    estimation_methods_avail: dict = {}
    _estimation_method = None
    cfg = None

    def __new__(cls, cfg: dict | None = None):
        """
        Return the plugin associated with the estimation_method given in the configuration

        :param cfg: configuration {'estimation_method': value}
        """
        if cls is AbstractEstimation:
            if isinstance(cfg["estimation_method"], str):
                try:
                    return super().__new__(cls.estimation_methods_avail[cfg["estimation_method"]])
                except KeyError:
                    logging.error("No estimation method named %s supported", cfg["estimation_method"])
                    raise KeyError
            else:
                if isinstance(cfg["estimation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super().__new__(cls.estimation_methods_avail[cfg["estimation_method"].encode("utf-8")])
                    except KeyError:
                        logging.error(
                            "No estimation method named %s supported",
                            cfg["estimation_method"],
                        )
                        raise KeyError
        else:
            return super().__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str) -> Callable[[type[Self]], type[Self]]:
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        """

        def decorator(subclass: type[Self]) -> type[Self]:
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
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
    def compute_estimation(self, img_left: xr.Dataset, img_right: xr.Dataset) -> tuple[dict, dict, np.ndarray, dict]:
        """
        Compute the phase cross correlation method

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
        :return:row disparity: Dict
                col disparity: Dict
                Calculated shifts: list
                Extra information about estimation : dict
        """

    @staticmethod
    def update_cfg_with_estimation(
        cfg: dict, disp_col: dict, disp_row: dict, shifts: np.ndarray, extra_dict: dict = None
    ) -> dict:
        """
        Save calculated shifts in a configuration dictionary

        :param cfg: user configuration
        :param disp_col: dict with init and range for disparity in column
        :param disp_row: dict with init and range for disparity in row
        :param shifts: computed global shifts between left and right
        :param extra_dict: Dictionary containing extra information about estimation
        :return: cfg: global configuration
        """

        cfg["input"]["col_disparity"] = disp_col
        cfg["input"]["row_disparity"] = disp_row

        cfg["pipeline"]["estimation"]["estimated_shifts"] = shifts.tolist()
        if extra_dict is not None:
            for key, value in extra_dict.items():
                cfg["pipeline"]["estimation"][key] = value

        return cfg
