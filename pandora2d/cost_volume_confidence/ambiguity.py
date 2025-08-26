# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the cost volume condifence computation step
with ambiguity method.
"""

import logging
from typing import Dict, Tuple
from json_checker import And

import numpy as np
import xarray as xr

from pandora2d.cost_volume_confidence.registry import CostVolumeConfidenceRegistry
from .cost_volume_confidence import CostVolumeConfidence


@CostVolumeConfidenceRegistry.add("ambiguity")
class Ambiguity(CostVolumeConfidence):
    """
    Ambiguity class
    """

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of Ambiguity class

        :param cfg: user_config for cost volume confidence
        :type cfg: dict
        :return: None
        """

        super().__init__(cfg)

        self._normalization = self._cfg["normalization"]
        self._eta_max = self._cfg["eta_max"]
        self._eta_step = self._cfg["eta_step"]

    @property
    def schema(self):
        return {
            "confidence_method": And(str, lambda x: x in ["ambiguity"]),
            "eta_max": And(float, lambda input: 0 < input < 1),
            "eta_step": And(float, lambda input: 0 < input < 1),
            "normalization": bool,
        }

    @property
    def defaults(self):
        return {
            "eta_max": 0.7,
            "eta_step": 0.01,
            "normalization": True,
        }

    def confidence_prediction(
        self,
        left_image: xr.Dataset,
        right_image: xr.Dataset,
        cost_volumes: xr.Dataset,
        dataset_disp_maps: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Compute a confidence prediction.

        :param left_image: left Dataset image
        :type left_image: xarray.Dataset
        :param right_image: right Dataset image
        :type right_image: xarray.Dataset
        :param cost_volumes: cost volume dataset
        :type cost_volumes: xarray.Dataset
        :param dataset_disp_maps: dataset containg row and col disparity maps
        :type dataset_disp_maps: xarray.Dataset
        :return: the disparity map and the cost volume updated with the confidence measure
        :rtype: Tuple[xr.Dataset, xr.Dataset]
        """

        logging.warning("The ambiguity method has not yet been implemented")

        # Fill confidence_measure data variables with zeros to test cost volume confidence ouput is correct
        if len(dataset_disp_maps.data_vars) != 0:
            confidence = xr.DataArray(
                np.zeros(
                    (len(dataset_disp_maps.row), len(dataset_disp_maps.col)),
                    dtype=dataset_disp_maps["row_map"].data.dtype,
                ),
                coords={"row": dataset_disp_maps.row, "col": dataset_disp_maps.col},
                dims=("row", "col"),
            )
            dataset_disp_maps["confidence_measure"] = confidence

        return cost_volumes, dataset_disp_maps
