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
Module for common base of all MatchingCost methods.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, cast, Mapping

import xarray as xr
from json_checker import And, Checker
from pandora.margins import Margins

import pandora2d.schema as cst_schema


class BaseMatchingCost(ABC):
    """MatchingCost base class."""

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        self._cfg = self.check_conf(cfg)

        self._matching_cost_method = self._cfg["matching_cost_method"]
        # Cast to int in order to help mypy because self.cfg is a Dict, and it can not know the type of step.
        self._step_row = cast(int, self._cfg["step"][0])
        self._step_col = cast(int, self._cfg["step"][1])
        self._window_size = cast(int, self._cfg["window_size"])
        self._subpix = cast(int, self._cfg["subpix"])
        # To move down if #226 use option 0
        self._spline_order = cast(int, self._cfg["spline_order"])

    @property
    def schema(self):
        return {
            # Census is not expected to be used with Pandora2D
            "matching_cost_method": And(str, lambda x: x not in ["census"]),
            "window_size": int,
            "step": cst_schema.STEP_SCHEMA,
            "spline_order": And(int, lambda y: 1 <= y <= 5),
            "subpix": And(int, lambda sp: sp in [1, 2, 4]),
        }

    @property
    def defaults(self):
        return {
            "window_size": 5,
            "subpix": 1,
            "step": [1, 1],
            # To move down if #226 use option 0
            "spline_order": 1,
        }

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """Check the matching cost configuration

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        updated_config = self._update_with_default_config_values(cfg)
        checker = Checker(self.schema)
        checker.validate(updated_config)

        return updated_config

    def _update_with_default_config_values(self, cfg: Dict):
        return {**self.defaults, **cfg}

    @property
    def cfg(self) -> Mapping[str, Union[str, int, List[int]]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        :rtype: cfg: dict
        """
        return self._cfg

    @property
    def _step(self) -> List[int]:
        """
        Get step [row, col]

        :return: step: list with row & col step
        :rtype: step: list
        """
        return [self._step_row, self._step_col]

    @property
    @abstractmethod
    def margins(self) -> Margins: ...

    @abstractmethod
    def allocate(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cfg: Dict,
        margins: Margins = None,
    ) -> None:
        """

        Allocate the cost volume

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :param margins: refinement margins
        :type margins: Margins
        :return: None
        """

    @abstractmethod
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
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param margins: refinement margins
        :type margins: Margins
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """
