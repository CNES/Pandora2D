# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Module for Dichotomy refinement method.
"""

import logging
from typing import Dict, Tuple, Union, NamedTuple, Callable

import numpy as np
import xarray as xr

from json_checker import And

from pandora2d.interpolation_filter import AbstractFilter
from . import refinement

COST_SELECTION_METHOD_MAPPING = {"min": np.nanargmin, "max": np.nanargmax}


@refinement.AbstractRefinement.register_subclass("dichotomy")
class Dichotomy(refinement.AbstractRefinement):
    """Subpixel refinement method by dichotomy."""

    NB_MAX_ITER = 9
    schema = {
        "refinement_method": And(str, lambda x: x in ["dichotomy"]),
        "iterations": And(int, lambda it: it > 0),
        "filter": And(str, lambda x: x in ["sinc", "bicubic"]),
    }

    _filter = None

    def __init__(self, cfg: dict = None, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        super().__init__(cfg)

        self._filter = AbstractFilter(  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated
            self.cfg["filter"]
        )

    @classmethod
    def check_conf(cls, cfg: Dict) -> Dict:
        """
        Check the refinement method configuration.

        Will change `number_of_iterations` value by `Dichotomy.NB_MAX_ITER` if above `Dichotomy.NB_MAX_ITER`.

        :param cfg: user_config for refinement method
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
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

        return self._filter.margins

    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixel disparity maps
        :type disp_map: xarray.Dataset
        :param img_left: left image dataset
        :type img_left: xarray.Dataset
        :param img_right: right image dataset
        :type img_right: xarray.Dataset
        :return: the refined disparity maps
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        cost_selection_method = COST_SELECTION_METHOD_MAPPING[cost_volumes.attrs["type_measure"]]

        # Initial disparity maps
        row_map = disp_map["row_map"]
        col_map = disp_map["col_map"]

        cost_surfaces = CostSurfaces(cost_volumes)

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
        cost_values = (
            cost_volumes["cost_volumes"]
            .sel(
                disp_row=row_map.where(~invalid_row_disparity_map_mask, cost_volumes.coords["disp_row"][0]),
                disp_col=col_map.where(~invalid_col_disparity_map_mask, cost_volumes.coords["disp_col"][0]),
            )
            .data
        )
        # Values are NaN if either row or column disparities are invalid
        invalid_disparity_map_mask = invalid_row_disparity_map_mask | invalid_col_disparity_map_mask
        cost_values[invalid_disparity_map_mask] = np.nan

        # row_disparity_source and col_disparity_sources contain the user disparity range
        row_disparity_source = cost_volumes.attrs["row_disparity_source"]
        col_disparity_source = cost_volumes.attrs["col_disparity_source"]

        precisions = [1 / 2 ** (it + 1) for it in range(self.cfg["iterations"])]

        # Convert disparity maps to np.array to optimise performance
        row_map = row_map.to_numpy()
        col_map = col_map.to_numpy()

        # See usage of np.nditer:
        # https://numpy.org/doc/stable/reference/arrays.nditer.html#modifying-array-values
        with np.nditer(
            [cost_values, row_map, col_map],
            op_flags=[["readwrite"], ["readwrite"], ["readwrite"]],
        ) as iterators:
            for cost_surface, (cost_value, disp_row_init, disp_col_init) in zip(cost_surfaces, iterators):

                # Invalid value
                if np.isnan(cost_value):
                    continue

                # If the best candidate found at the disparity step is at the edge of the disparity range
                # we do no enter the dichotomy loop
                if (disp_row_init in row_disparity_source) or (disp_col_init in col_disparity_source):
                    continue

                # pos_disp_col_init corresponds to the position in the cost surface
                # of the initial disparity value in column
                # In cost surface, disp_col are along rows, so pos_disp_col_init corresponds to a row index
                pos_disp_col_init = cost_surface["disp_col"].searchsorted(disp_col_init)

                # pos_disp_row_init corresponds to the position in the cost surface
                # of the initial disparity value in row
                # In cost surface, disp_row are along columns, so pos_disp_row_init corresponds to a column index
                pos_disp_row_init = cost_surface["disp_row"].searchsorted(disp_row_init)

                ###         Example:        ###

                # a cost surface is shown below:
                #
                #                  disp_row
                #               -2 -1 0  1  2
                #
                #          -2   0  1  2  3  4
                #          -1   0  1  2  3  4
                #  disp_col 0   0  1  2  3  4
                #           1   0  6  2  3  4
                #           2   0  1  2  3  4
                #
                # Here the best coefficient is 6 (max cost selection).
                # This coefficient corresponds to:
                # disp_row_init = -1
                # disp_col_init = 1
                # pos_disp_row_init = 1
                # pos_disp_col_init = 3

                for precision in precisions:
                    # Syntax disp_row_init[...] is for assign value back to row_map with np.nditer
                    (pos_disp_row_init, pos_disp_col_init), disp_row_init[...], disp_col_init[...], cost_value[...] = (
                        search_new_best_point(
                            cost_surface.data,
                            precision,
                            (disp_row_init, disp_col_init),  # type: ignore # Reason: is 0 dim array
                            (pos_disp_row_init, pos_disp_col_init),
                            cost_value,  # type: ignore # Reason: is 0 dim array
                            self._filter,
                            cost_selection_method,
                        )
                    )

        logging.info("Dichotomy precision reached: %s", precisions[-1])

        return col_map, row_map, cost_values


class CostSurfaces:
    """
    Container to extract subsampling cost surfaces around a given disparity from cost volumes.

    Cost Surface of point with coordinates `row==0` and `col==1` can be accessed with   `cost_surface[0, 1]`.

    The container is iterable row first then columns.
    """

    def __init__(self, cost_volumes: xr.Dataset):
        """
        Extract subsampling cost surfaces from cost volumes around a given disparity from cost volumes.

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixels disparity maps
        :param disparity_margins: margins used to define disparity ranges
        :type disparity_margins: Margins
        """
        self.cost_volumes = cost_volumes

    def __getitem__(self, item):
        """Get cost surface of coordinates item where item is (row, col)."""
        row, col = item
        return self.cost_volumes["cost_volumes"].sel(row=row, col=col)

    def __iter__(self):
        """Iter over cost surfaces, row first then columns."""
        for row in self.cost_volumes.coords["row"].data:
            for col in self.cost_volumes.coords["col"].data:
                yield self[row, col]


class Point(NamedTuple):
    """Coordinates of a subpixellic point of cost surface."""

    row: Union[int, float, np.float32]
    col: Union[int, float, np.float32]


def all_same(sequence):
    """Return True if all items in sequence are equals."""
    return len(set(sequence)) == 1


def search_new_best_point(
    cost_surface: np.ndarray,
    precision: float,
    initial_disparity: Union[Tuple[np.floating, np.floating], Tuple[int, int]],
    initial_position: Union[Tuple[np.floating, np.floating], Tuple[int, int]],
    initial_value: np.float32,
    filter_dicho: AbstractFilter,
    cost_selection_method: Callable,
) -> Tuple[Point, np.floating, np.floating, np.floating]:
    """
    Find best position and cost after interpolation of cost surface for given precision.

    :param cost_surface: Disparities in rows and cols of a point
    :type cost_surface: np.ndarray
    :param precision: subpixellic precision to use
    :type precision: float
    :param initial_disparity: initial disparities (disp_row, disp_col)
    :type initial_disparity: Union[Tuple[np.floating, np.floating], Tuple[int, int]]
    :param initial_position: coordinates (row, col) to interpolate around
    :type initial_position: Union[Tuple[np.floating, np.floating], Tuple[int, int]]
    :param initial_value: initial value
    :type initial_value: np.float32
    :param filter_dicho: filter used to do interpolation in dichotomy loop
    :type filter_dicho: AbstractFilter
    :param cost_selection_method: function used to select best cost
    :type cost_selection_method: Callable
    :return: coordinates of best interpolated cost, its value and its corresponding disparities.
    :rtype: Tuple[Point, np.floating, np.floating, np.floating]
    """

    # initial_disp_row corresponds to row_map[i,j]
    # initial_disp_col corresponds to col_map[i,j]
    initial_disp_row, initial_disp_col = initial_disparity

    # initial_pos_disp_row is the column index corresponding to initial_disp_row in cost surface
    # initial_pos_disp_col is the row index corresponding to initial_disp_col in cost surface
    initial_pos_disp_row, initial_pos_disp_col = initial_position

    # Used to compute new positions and new disparities based on precision
    disp_row_shifts = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float32) * precision
    disp_col_shifts = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float32) * precision

    # disp_row are along columns in cost_surface, then new_cols are computed from initial_pos_disp_row
    new_cols = disp_row_shifts + initial_pos_disp_row
    # disp_col are along rows in cost_surface, then new_rows are computed from initial_pos_disp_col
    new_rows = disp_col_shifts + initial_pos_disp_col

    # New subpixel disparity values
    new_rows_disp = disp_row_shifts + initial_disp_row
    new_cols_disp = disp_col_shifts + initial_disp_col

    # Interpolate points at positions (new_rows[i], new_cols[i])
    candidates = filter_dicho.interpolate(cost_surface, (new_cols, new_rows))

    # In case a NaN is present in the kernel, candidates will be all-NaNs. Let’s restore initial_position value so
    # that best candidate search will be able to find it.
    candidates[4] = float(initial_value)

    # If all interpolated coefficients are equals,
    # we keep the initial disparity as the best candidate
    initial_best_index = 4
    best_index = initial_best_index if all_same(candidates) else cost_selection_method(candidates)

    # We return:
    # - index of best new disparities in cost surface
    # - new best disparities
    # - value of best similarity coefficient for new best disparities
    return (
        Point(new_cols[best_index], new_rows[best_index]),
        new_rows_disp[best_index],
        new_cols_disp[best_index],
        candidates[best_index],
    )
