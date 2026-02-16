# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Module for common base of all cost volume confidence methods.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import xarray as xr
from json_checker import Checker


class CostVolumeConfidence(ABC):
    """CostVolumeConfidence base class."""

    def __init__(self, cfg: dict) -> None:
        """
        Initialisation of CostVolumeConfidence class

        :param cfg: user_config for cost volume confidence
        :return: None
        """
        self._cfg = self.check_conf(cfg)
        self._method = cfg["confidence_method"]

    @property
    @abstractmethod
    def schema(self):
        """
        Configuration schema
        """

    @property
    @abstractmethod
    def defaults(self):
        """
        Configuration default values
        """

    def check_conf(self, cfg: dict) -> dict[str, str]:
        """
        Check the cost volume confidence configuration

        :param cfg: user_config for cost volume confidence
        :return: cfg: global configuration
        """
        updated_config = self._update_with_default_config_values(cfg)
        checker = Checker(self.schema)
        checker.validate(updated_config)

        return updated_config

    def _update_with_default_config_values(self, cfg: dict):
        return {**self.defaults, **cfg}

    @property
    def cfg(self) -> Mapping[str, str | int | list[int]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        """
        return self._cfg

    @abstractmethod
    def confidence_prediction(
        self,
        left_image: xr.Dataset,
        right_image: xr.Dataset,
        cost_volumes: xr.Dataset,
        dataset_disp_maps: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Compute a confidence prediction.

        :param left_image: left Dataset image
        :param right_image: right Dataset image
        :param cost_volumes: cost volume dataset
        :param dataset_disp_maps: dataset containg row and col disparity maps
        :return: the disparity map and the cost volume updated with the confidence measure
        """
