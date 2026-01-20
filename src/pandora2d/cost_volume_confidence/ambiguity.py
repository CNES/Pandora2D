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

import logging

import numpy as np
import xarray as xr
from json_checker import And
from pandora import cost_volume_confidence as pandora_confidence
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
        # en colonne uniquement
        etas = np.arange(self._eta_min, self._eta_max, self._eta_step)
        nbr_etas = etas.shape[0]
        grids = left_image.col_disparity
        disparity_range_col = cost_volumes.disp_col

        cv_4d = cost_volumes["cost_volumes"].data
        cv_3d = cv_4d.reshape(
            cost_volumes.sizes["row"],
            cost_volumes.sizes["col"],
            cost_volumes.sizes["disp_row"] * cost_volumes.sizes["disp_col"],
        )

        ambiguity_ = pandora_confidence.AbstractCostVolumeConfidence(
            **{
                "confidence_method": "ambiguity",
                "eta_max": self._eta_max,
                "eta_step": self._eta_step,
                "normalization": False,
            }
        )

        ambiguity = ambiguity_.compute_ambiguity(cv_3d, etas, nbr_etas, grids, disparity_range_col)

        if self._normalization:
            if "global_disparity" in left_image.attrs:
                ambiguity = self.normalize_with_extremum(
                    ambiguity, left_image, nbr_etas=nbr_etas, subpix=cost_volumes.attrs["subpixel"]
                )
                logging.info(
                    "You are not using ambiguity normalization by percentile; \n"
                    "you are in a specific case with the instantiation of global_disparity."
                )
            # in case of cross correlation
            elif "global_disparity" in right_image.attrs:
                ambiguity = self.normalize_with_extremum(
                    ambiguity, right_image, nbr_etas=nbr_etas, subpix=cost_volumes.attrs["subpixel"]
                )
            else:
                ambiguity = self.normalize_with_percentile(ambiguity)

        # Conversion of ambiguity into a confidence measure
        ambiguity = 1 - ambiguity

        # Fill confidence_measure data variables with zeros to test cost volume confidence output is correct
        if len(dataset_disp_maps.data_vars) != 0:
            logging.info("save ambiguity in dataset")
            confidence = xr.DataArray(
                ambiguity,
                coords={"row": dataset_disp_maps.row, "col": dataset_disp_maps.col},
                dims=("row", "col"),
            )
            dataset_disp_maps["confidence_measure"] = confidence

        return cost_volumes, dataset_disp_maps

    def normalize_with_percentile(self, ambiguity: np.ndarray) -> np.ndarray:
        """
        Normalize ambiguity with percentile .
        Cost Volume must correspond to min similarity measure

        :param ambiguity: ambiguity
        :type ambiguity: 2D np.ndarray (row, col) dtype = float32
        :return: the normalized ambiguity
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """

        norm_amb = np.copy(ambiguity)
        perc_min = np.percentile(norm_amb, self._percentile)
        perc_max = np.percentile(norm_amb, 100 - self._percentile)
        np.clip(norm_amb, perc_min, perc_max, out=norm_amb)

        return (norm_amb - np.min(norm_amb)) / (np.max(norm_amb) - np.min(norm_amb))

    @staticmethod
    def normalize_with_extremum(
        confidence: np.ndarray, dataset: xr.Dataset, nbr_etas: int, subpix: int = 1
    ) -> np.ndarray:
        """
        Normalize ambiguity with extremum

        :param confidence: confidence
        :type confidence: 2D np.ndarray (row, col) dtype = float32
        :param dataset: Dataset image
        :tye dataset: xarray.Dataset
        :param nbr_etas: size of etas
        :type nbr_etas: int
        :param subpix:  subpix used in matching cost
        :type subpix: int
        :return: the normalized confidence
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """
        norm_confidence = np.copy(confidence)
        global_disp_max = dataset.attrs["global_disparity"][1]
        global_disp_min = dataset.attrs["global_disparity"][0]
        max_norm = (global_disp_max - global_disp_min) * nbr_etas * subpix

        return norm_confidence / max_norm
