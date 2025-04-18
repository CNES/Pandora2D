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
from typing import Dict, Tuple
from abc import abstractmethod, ABCMeta
from json_checker import Checker

import xarray as xr
import numpy as np

from pandora.margins.descriptors import NullMargins


class AbstractRefinement:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    refinement_methods_avail: Dict = {}
    _refinement_method = None
    margins = NullMargins()

    schema: Dict  # This will raise an AttributeError if not override in subclasses

    # If we don't make cfg optional, we got this error when we use subprocesses in refinement_method :
    # AbstractRefinement.__new__() missing 1 required positional argument: 'cfg'
    def __new__(cls, cfg: dict = None, _: list = None, __: int = 5):
        """
        Return the plugin associated with the refinement_method given in the configuration

        :param cfg: configuration {'refinement_method': value}
        :type cfg: dictionary
        """

        if cls is AbstractRefinement:
            if isinstance(cfg["refinement_method"], str):
                try:
                    return super(AbstractRefinement, cls).__new__(
                        cls.refinement_methods_avail[cfg["refinement_method"]]
                    )
                except KeyError:
                    logging.error("No subpixel method named %s supported", cfg["refinement_method"])
                    raise KeyError
            else:
                if isinstance(cfg["refinement_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractRefinement, cls).__new__(
                            cls.refinement_methods_avail[cfg["refinement_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No subpixel method named %s supported",
                            cfg["refinement_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractRefinement, cls).__new__(cls)
        return None

    def desc(self) -> None:
        """
        Describes the refinement method
        :return: None
        """
        print(f"{self._refinement_method} refinement measure")

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
            cls.refinement_methods_avail[short_name] = subclass
            return subclass

        return decorator

    def __init__(self, cfg: Dict, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :param step: list containing row and col step
        :type step: list
        :param window_size: window size
        :type window_size: int
        :return: None
        """
        self.cfg = self.check_conf(cfg)

    @classmethod
    def check_conf(cls, cfg: Dict) -> Dict:
        """
        Check the refinement method configuration.

        :param cfg: user_config for refinement method
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        checker = Checker(cls.schema)
        checker.validate(cfg)

        return cfg

    @abstractmethod
    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixels disparity maps
        :type disp_map: xarray.Dataset
        :param img_left: left image dataset
        :type img_left: xarray.Dataset
        :param img_right: right image dataset
        :type img_right: xarray.Dataset
        :return: the refined disparity maps
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
