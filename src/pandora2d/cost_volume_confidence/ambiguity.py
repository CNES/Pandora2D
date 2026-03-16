# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the cost volume confidence computation step
with ambiguity method.
"""

import numpy as np
import xarray as xr
from json_checker import And

from pandora.cost_volume_confidence.ambiguity import Ambiguity as pandora_ambiguity
from pandora2d.cost_volume_confidence.registry import CostVolumeConfidenceRegistry

from .cost_volume_confidence import CostVolumeConfidence


@CostVolumeConfidenceRegistry.add("ambiguity")
class Ambiguity(CostVolumeConfidence):
    """
    Ambiguity class
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialisation of Ambiguity class

        :param cfg: user_config for cost volume confidence
        :return: None
        """

        super().__init__(cfg)

        self._normalization = self._cfg["normalization"]
        self._eta_min = 0.0
        self._eta_max = self._cfg["eta_max"]
        self._eta_step = self._cfg["eta_step"]
        self._percentile = 1

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
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Compute a confidence prediction.

        :param left_image: left Dataset image
        :param right_image: right Dataset image
        :param cost_volumes: cost volume dataset
        :param dataset_disp_maps: dataset containing row and col disparity maps
        :return: the disparity map and the cost volume updated with the confidence measure
        """
        # Using Pandora to perform calculations on columns only
        etas = np.arange(self._eta_min, self._eta_max, self._eta_step)  # type: np.ndarray
        nbr_etas = etas.shape[0]
        grids = left_image.col_disparity
        disparity_range_col = cost_volumes.disp_col
        nbr_disparities = cost_volumes.sizes["disp_row"] * cost_volumes.sizes["disp_col"]

        # Reverse cost_volume if matching_cost measure is "max"
        type_measure_max = cost_volumes.attrs["type_measure"] == "max"
        if type_measure_max:
            cost_volumes["cost_volumes"].data *= -1

        cost_volumes_4d = cost_volumes["cost_volumes"].data
        cost_volumes_3d = cost_volumes_4d.reshape(
            cost_volumes.sizes["row"],
            cost_volumes.sizes["col"],
            nbr_disparities,
        )

        ambiguity_ = pandora_ambiguity(
            confidence_method="ambiguity",
            eta_max=self._eta_max,
            eta_step=self._eta_step,
            normalization=False,
        )

        ambiguity = ambiguity_.compute_ambiguity(cost_volumes_3d, etas, nbr_etas, grids, disparity_range_col)

        if self._normalization:
            ambiguity = self.normalize_with_extremum(ambiguity, nbr_disparities, nbr_etas)

        # Conversion of ambiguity into a confidence measure
        confidence_measure = 1 - ambiguity

        # Fill confidence_measure data variables with zeros to test cost volume confidence output is correct
        confidence = xr.DataArray(
            confidence_measure,
            coords={"row": dataset_disp_maps.row, "col": dataset_disp_maps.col},
            dims=("row", "col"),
        )
        dataset_disp_maps["confidence_measure"] = confidence

        # Remove modification
        if type_measure_max:
            cost_volumes["cost_volumes"].data *= -1

        return cost_volumes, dataset_disp_maps

    @staticmethod
    def normalize_with_extremum(confidence: np.ndarray, nbr_disparities: int, nbr_etas: int) -> np.ndarray:
        """
        Normalize ambiguity with extremum

        :param confidence: confidence
        :param nbr_disparities: number of disparity (row_disparity * col_disparity)
        :param nbr_etas: size of etas
        :return: the normalized confidence
        """
        max_norm = nbr_disparities * nbr_etas
        return confidence / max_norm
