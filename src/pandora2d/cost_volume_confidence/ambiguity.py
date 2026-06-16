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
from pandora2d.common import get_cost_volume_without_margins
from pandora2d.cost_volume_confidence.registry import CostVolumeConfidenceRegistry
from pandora2d.margins import Margins

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

        # Reverse cost_volume if matching_cost measure is "max"
        type_measure_max = cost_volumes.attrs["type_measure"] == "max"
        if type_measure_max:
            cost_volumes["cost_volumes"].data *= -1

        # Check margins presence
        disparity_margins = cost_volumes.attrs["disparity_margins"]
        if disparity_margins is not None and disparity_margins != Margins(0, 0, 0, 0):
            cost_volumes_to_use = get_cost_volume_without_margins(cost_volumes)
        else:
            cost_volumes_to_use = cost_volumes

        # Using Pandora to perform calculations on columns only
        # Cast to float to avoid mypy error
        etas = np.arange(self._eta_min, float(self._eta_max), float(self._eta_step))  # type: np.ndarray
        nbr_etas = etas.shape[0]
        nbr_row = cost_volumes_to_use.sizes["row"]
        nbr_col = cost_volumes_to_use.sizes["col"]
        nbr_disparities = cost_volumes_to_use.sizes["disp_row"] * cost_volumes_to_use.sizes["disp_col"]
        disparity_range_col = cost_volumes_to_use.disp_col
        cost_volumes_4d = cost_volumes_to_use["cost_volumes"].data
        grids = left_image.col_disparity

        # Reshape cost_volume 4D into cost_volume 3D (row, col, disp_row*disp_col) to use pandora ambiguity
        cost_volumes_3d = cost_volumes_4d.reshape(nbr_row, nbr_col, nbr_disparities)

        # Init pandora ambiguity instance
        ambiguity_ = pandora_ambiguity(
            confidence_method="ambiguity",
            eta_max=self._eta_max,
            eta_step=self._eta_step,
            normalization=False,
        )

        # Compute ambiguity
        ambiguity = ambiguity_.compute_ambiguity(cost_volumes_3d, etas, nbr_etas, grids, disparity_range_col)

        if self._normalization:
            ambiguity = self.normalize_with_extremum(ambiguity, nbr_disparities, nbr_etas)

        # Conversion of ambiguity into a confidence measure
        # Please note: this creates a new data structure the size of an image, which increases memory usage
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
