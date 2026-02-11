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
"""
Module for Dichotomy refinement method (cpp version).
"""

import logging

import numpy as np
import xarray as xr
from json_checker import And

from ..constants import Criteria
from ..interpolation_filter import AbstractFilter
from ..refinement_cpp import refinement_bind
from . import refinement


@refinement.AbstractRefinement.register_subclass("dichotomy")
class Dichotomy(refinement.AbstractRefinement):
    """Subpixel refinement method by dichotomy (cpp version)."""

    NB_MAX_ITER = 9
    schema = {
        "refinement_method": And(str, lambda x: x in ["dichotomy"]),
        "iterations": And(int, lambda it: it > 0),
        "filter": And(dict, lambda method: method["method"] in AbstractFilter.interpolation_filter_methods_avail),
    }

    def __init__(self, cfg: dict = None, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :return: None
        """

        super().__init__(cfg)
        fractional_shift_ = 2 ** -self.cfg["iterations"]
        self.filter = AbstractFilter(  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated
            self.cfg["filter"], fractional_shift=fractional_shift_
        )

    @classmethod
    def check_conf(cls, cfg: dict) -> dict:
        """
        Check the refinement method configuration.

        Will change `number_of_iterations` value by `Dichotomy.NB_MAX_ITER` if above `Dichotomy.NB_MAX_ITER`.

        :param cfg: user_config for refinement method
        :return: cfg: global configuration
        """
        cfg = super().check_conf(cfg)
        if cfg["iterations"] > cls.NB_MAX_ITER:
            logging.warning(
                "number_of_iterations %s is above maximum iteration. Maximum value of %s will be used instead.",
                cfg["iterations"],
                cls.NB_MAX_ITER,
            )
            cfg["iterations"] = cls.NB_MAX_ITER
        return cfg

    @property
    def margins(self):
        """
        Create margins for dichotomy object.

        It will be used for ROI and for dichotomy window extraction from cost volumes.
        """

        return self.filter.margins

    def refinement_method(  # pylint: disable=too-many-locals
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :param disp_map: pixel disparity maps
        :param img_left: left image dataset
        :param img_right: right image dataset
        :return: the refined disparity maps
        """

        # Initial disparity maps
        row_map = disp_map["row_map"]
        col_map = disp_map["col_map"]

        # Get score map
        cost_values = create_cost_values_map(cost_volumes, disp_map)

        criteria_map = (
            (disp_map["validity"].sel(criteria="partial_validity_mask") == 1).astype(int)  # select invalids
            | disp_map["validity"].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name)
        ).data

        subpixel = cost_volumes.attrs["subpixel"]
        cost_volume_type = cost_volumes["cost_volumes"].data.dtype

        # Convert disparity maps to np.array to optimise performance
        # Transforming disparity maps into index maps
        row_map = disparity_to_index(row_map, cost_volumes.disp_row.values[0], subpixel).astype(cost_volume_type)
        col_map = disparity_to_index(col_map, cost_volumes.disp_col.values[0], subpixel).astype(cost_volume_type)

        if np.issubdtype(cost_volume_type, np.float32):
            compute_dichotomy = refinement_bind.compute_dichotomy_float
        elif np.issubdtype(cost_volume_type, np.float64):
            compute_dichotomy = refinement_bind.compute_dichotomy_double
        else:
            raise TypeError("Cost volume must be in np.float32 or np.float64")

        compute_dichotomy(
            cost_volumes.cost_volumes.data,
            col_map.ravel(),
            row_map.ravel(),
            cost_values.ravel(),
            criteria_map.ravel(),
            cost_volumes.criteria.data,
            subpixel,
            self.cfg["iterations"],
            self.filter.cpp_instance,
            cost_volumes.attrs["type_measure"],
        )

        # Inverse transforming index maps into disparity maps
        col_map = index_to_disparity(col_map, cost_volumes.disp_col.values[0], subpixel)
        row_map = index_to_disparity(row_map, cost_volumes.disp_row.values[0], subpixel)

        # Log about precision
        subpixel_to_iteration = cost_volumes.attrs["subpixel"].bit_length() - 1
        precision = 1 / 2 ** max(self.cfg["iterations"], subpixel_to_iteration)
        logging.info("Dichotomy precision reached: %s", precision)

        return col_map, row_map, cost_values


def disparity_to_index(disparity_map: xr.DataArray, shift: int, subpixel: int) -> np.ndarray:
    """
    Transform a disparity map to index map. Indexes correspond to (row/col) disparities in cost volume.

    Example:
        - with subpixel=1 :

            * disparity_map = -2 -1 -1  1
                              -1  0 -1 -1
                               0  1  1  1

            * disparities range = [-4 -3 -2 -1 0 1 2 3]

            * index_map = 2 3 3 5
                          3 4 3 3
                          4 5 5 5

        - with subpixel=2 :

            * disparity_map = -4  -2   -1.5 -2.5
                              -4  -1   -1   -1
                              -4  -1.5 -1   -1.5

            * disparities range = [-4 -3.5 -3 -2.5 -2 -1.5 -1]

            * index_map = 0 4 5 3
                          0 6 6 6
                          0 5 6 5

    :param disparity_map: 2D map
    :param shift: the first value of the disparity coordinates in the cost volume
    :param subpixel: :sub-sampling of cost_volume
    :return: the index map
    """
    return (disparity_map.to_numpy() - shift) * subpixel


def index_to_disparity(index_map: np.ndarray, shift: int, subpixel: int) -> np.ndarray:
    """
    Transform an index map to disparity map. Indexes correspond to (row/col) disparities in cost volume.

    For examples, see disparity_to_index method.

    :param index_map: 2D map
    :param shift: the first value of the disparity coordinates in the cost volume
    :param subpixel: :sub-sampling of cost_volume
    :return: the index map
    """
    return (index_map / subpixel) + shift


def create_cost_values_map(cost_volumes: xr.Dataset, disp_map: xr.Dataset) -> np.ndarray:
    """
    Return the map with best matching score

    :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
    :param disp_map: pixel disparity maps
    :return: the cost_value map and the invalid_disparity_map
    """
    # Initial disparity maps
    row_map = disp_map["row_map"]
    col_map = disp_map["col_map"]

    # With invalid data in disparity maps, there is no corresponding data in cost volume, so we temporarily
    # replace them by an existing value to permits the extraction, then we put NaNs at corresponding coordinates
    # in values dataset.
    invalid_disparity = disp_map.attrs["invalid_disp"]
    invalid_row_disparity_map_mask: np.ndarray = (
        row_map.isnull().data if np.isnan(invalid_disparity) else row_map.isin(invalid_disparity).data
    )
    invalid_col_disparity_map_mask: np.ndarray = (
        col_map.isnull().data if np.isnan(invalid_disparity) else col_map.isin(invalid_disparity).data
    )
    return cost_volumes.cost_volumes.sel(
        disp_row=row_map.where(~invalid_row_disparity_map_mask, cost_volumes.coords["disp_row"][0]),
        disp_col=col_map.where(~invalid_col_disparity_map_mask, cost_volumes.coords["disp_col"][0]),
    ).data
