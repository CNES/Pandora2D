#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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

import logging
from typing import Dict, Tuple
from abc import abstractmethod, ABCMeta

import xarray as xr
import numpy as np


class AbstractRefinement:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    refinement_methods_avail: Dict = {}
    _refinement_method = None
    cfg = None

    def __new__(cls, **cfg: dict):
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
                    logging.error("No subpixel method named % supported", cfg["refinement_method"])
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
                            "No subpixel method named % supported",
                            cfg["refinement_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractRefinement, cls).__new__(cls)
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
            cls.refinement_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def refinement_method(self, cost_volumes: xr.Dataset, pixel_maps: xr.Dataset) -> Tuple[np.array, np.array]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.dataset
        :param pixel_maps: pixels disparity maps
        :type pixel_maps: xarray.dataset
        :return: the refined disparity maps
        :rtype: Tuple[np.array, np.array]
        """
