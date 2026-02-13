#!/usr/bin/env python
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the refinement computation step.
"""

from __future__ import annotations

import logging
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np
import xarray as xr
from json_checker import Checker

from pandora2d.margins import NullMargins

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class AbstractRefinement:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    refinement_methods_avail: dict = {}
    _refinement_method = None

    schema: dict  # This will raise an AttributeError if not override in subclasses

    # If we don't make cfg optional, we got this error when we use subprocesses in refinement_method :
    # AbstractRefinement.__new__() missing 1 required positional argument: 'cfg'
    def __new__(cls, cfg: dict = None, _: list = None, __: int = 5):
        """
        Return the plugin associated with the refinement_method given in the configuration

        :param cfg: configuration {'refinement_method': value}
        """

        if cls is AbstractRefinement:
            if isinstance(cfg["refinement_method"], str):
                try:
                    return super().__new__(cls.refinement_methods_avail[cfg["refinement_method"]])
                except KeyError:
                    logging.error("No subpixel method named %s supported", cfg["refinement_method"])
                    raise KeyError
            else:
                if isinstance(cfg["refinement_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super().__new__(cls.refinement_methods_avail[cfg["refinement_method"].encode("utf-8")])
                    except KeyError:
                        logging.error(
                            "No subpixel method named %s supported",
                            cfg["refinement_method"],
                        )
                        raise KeyError
        else:
            return super().__new__(cls)
        return None

    def desc(self) -> None:
        """
        Describes the refinement method
        :return: None
        """
        print(f"{self._refinement_method} refinement measure")

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
            cls.refinement_methods_avail[short_name] = subclass
            return subclass

        return decorator

    def __init__(self, cfg: dict, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :param step: list containing row and col step
        :param window_size: window size
        :return: None
        """
        self.cfg = self.check_conf(cfg)

    @property
    def margins(self):
        """Return refinement's Margins."""
        return NullMargins()

    @classmethod
    def check_conf(cls, cfg: dict) -> dict:
        """
        Check the refinement method configuration.

        :param cfg: user_config for refinement method
        :return: cfg: global configuration
        """
        checker = Checker(cls.schema)
        checker.validate(cfg)

        return cfg

    @abstractmethod
    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :param disp_map: pixels disparity maps
        :param img_left: left image dataset
        :param img_right: right image dataset
        :return: the refined disparity maps
        """
