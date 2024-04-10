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
from functools import partial
from typing import Dict, Tuple, Union, NamedTuple, Callable

import numpy as np
import scipy
import xarray as xr

from json_checker import And

from pandora.margins import Margins
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
        return Margins(2, 2, 2, 2)  # Hard coded for map_coordinates of order 3.

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
        filter_method = partial(scipy.ndimage.map_coordinates, mode="mirror", prefilter=True, output=np.float32)
        cost_selection_method = COST_SELECTION_METHOD_MAPPING[cost_volumes.attrs["type_measure"]]

        row_map = disp_map["row_map"]
        col_map = disp_map["col_map"]
        # Because in cost_volumes, disparities in columns are along array rows and disparities in rows are along
        # array columns, `col_positions` use `up` margins and `row_positions` use `left` margins.
        # see test_dichotomy.py::TestDichotomyWindows::test_cost_volumes_dimensions_order
        col_positions = np.full_like(col_map, self.margins.up, dtype=np.float32)
        row_positions = np.full_like(row_map, self.margins.left, dtype=np.float32)
        # Due to filter footprint, we need margins to extract surrounding disparities around pixellic best
        # candidate.
        dichotomy_windows = DichotomyWindows(cost_volumes, disp_map, self.margins)

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

        precisions = [1 / 2 ** (it + 1) for it in range(self.cfg["iterations"])]

        # See usage of np.nditer:
        # https://numpy.org/doc/stable/reference/arrays.nditer.html#modifying-array-values
        with np.nditer(
            [cost_values, row_positions, col_positions], op_flags=[["readwrite"], ["readwrite"], ["readwrite"]]
        ) as iterators:
            for dichotomy_window, (cost_value, row_position, col_position) in zip(dichotomy_windows, iterators):
                if np.isnan(cost_value):
                    continue
                for precision in precisions:
                    # Syntax row_position[...] is for assign value back to row_positions with np.nditer
                    (row_position[...], col_position[...]), cost_value[...] = search_new_best_point(
                        dichotomy_window.data,
                        precision,
                        (row_position, col_position),  # type: ignore # Reason: is 0 dim array
                        cost_value,  # type: ignore # Reason: is 0 dim array
                        filter_method,
                        cost_selection_method,
                    )

        logging.info("Dichotomy precision reached: %s", precisions[-1])
        # Because disparities in columns are along array rows and disparities in rows are along array columns,
        # `delta_col` uses `row_positions` and `delta_row` uses `col_position`.
        delta_row = col_positions - self.margins.up
        delta_col = row_positions - self.margins.left

        new_row_map = build_subpixellic_disparity_map(
            delta_row,
            disp_map["row_map"],
            invalid_disparity,
            invalid_row_disparity_map_mask,
            *cost_volumes.attrs["row_disparity_source"],
        )
        new_col_map = build_subpixellic_disparity_map(
            delta_col,
            disp_map["col_map"],
            invalid_disparity,
            invalid_col_disparity_map_mask,
            *cost_volumes.attrs["col_disparity_source"],
        )
        return new_col_map, new_row_map, cost_values


def build_subpixellic_disparity_map(
    delta: np.ndarray,
    disp_map: xr.DataArray,
    invalid_disparity: Union[int, float],
    invalid_disparity_map_mask: np.ndarray,
    min_disparity: Union[int, float],
    max_disparity: Union[int, float],
) -> np.ndarray:
    """
    Build subpixellic map by applying delta to original one and clipping values to disparity range.

    :param delta:
    :type delta: np.ndarray
    :param disp_map:
    :type disp_map: xr.DataArray
    :param invalid_disparity:
    :type invalid_disparity: Union[int, float]
    :param invalid_disparity_map_mask:
    :type invalid_disparity_map_mask: np.ndarray
    :param min_disparity:
    :type min_disparity: Union[int, float]
    :param max_disparity:
    :type max_disparity: Union[int, float]
    :return: disparity map
    :rtype: np.ndarray
    """
    # Compute new maps taking subpixellic delta into account
    new_map = (disp_map + delta).data
    before_range = new_map < min_disparity
    after_range = new_map > max_disparity
    # Clip values to disparity range
    new_map[before_range] = min_disparity
    new_map[after_range] = max_disparity
    # clip operation removed invalid values, so let’s put them back
    new_map[invalid_disparity_map_mask] = invalid_disparity
    return new_map


class DichotomyWindows:
    """
    Container to extract subsampling cost surfaces around a given disparity from cost volumes.

    Dichotomy Window of point with coordinates `row==0` and `col==1` can be accessed with `dichotomy_window[0, 1]`.

    The container is iterable row first then columns.
    """

    def __init__(self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, disparity_margins: Margins):
        """
        Extract subsampling cost surfaces from cost volumes around a given disparity from cost volumes.

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixels disparity maps
        :param disparity_margins: margins used to define disparity ranges
        :type disparity_margins: Margins
        """
        self.cost_volumes = cost_volumes
        self.min_row_disp_map = disp_map["row_map"] - disparity_margins.up
        self.max_row_disp_map = disp_map["row_map"] + disparity_margins.down
        self.min_col_disp_map = disp_map["col_map"] - disparity_margins.left
        self.max_col_disp_map = disp_map["col_map"] + disparity_margins.right

    def __getitem__(self, item):
        """Get cost surface of coordinates item where item is (row, col)."""
        row, col = item
        row_slice = np.s_[self.min_row_disp_map.sel(row=row, col=col) : self.max_row_disp_map.sel(row=row, col=col)]
        col_slice = np.s_[self.min_col_disp_map.sel(row=row, col=col) : self.max_col_disp_map.sel(row=row, col=col)]
        return self.cost_volumes["cost_volumes"].sel(row=row, col=col, disp_row=row_slice, disp_col=col_slice)

    def __iter__(self):
        """Iter over cost surfaces, row first then columns."""
        for row in self.cost_volumes.coords["row"].data:
            for col in self.cost_volumes.coords["col"].data:
                yield self[row, col]


class Point(NamedTuple):
    """Coordinates of a subpixellic point of cost surface."""

    row: Union[int, float, np.float32]
    col: Union[int, float, np.float32]


def search_new_best_point(
    dichotomy_window: np.ndarray,
    precision: float,
    initial_position: Union[Tuple[np.floating, np.floating], Tuple[int, int]],
    initial_value: np.float32,
    filter_method: Callable,
    cost_selection_method: Callable,
) -> Tuple[Point, np.floating]:
    """
    Find best position and cost after interpolation of cost surface for given precision.

    :param dichotomy_window: Disparities in rows and cols of a point
    :type dichotomy_window: np.ndarray
    :param precision: subpixellic precision to use
    :type precision: float
    :param initial_position: coordinates (row, col) to interpolate around
    :type initial_position: Union[Tuple[np.floating, np.floating], Tuple[int, int]]
    :param initial_value: initial value
    :type initial_value: np.float32
    :param filter_method: function used to do interpolation. Its parameters should be:
                            - inputs: ndarray with values to interpolate from.
                            - coordinates: The coordinates at which input is evaluated.
    :type filter_method: Callable
    :param cost_selection_method: function used to select best cost
    :type cost_selection_method: Callable
    :return: coordinates of best interpolated cost and its value.
    :rtype: Tuple[Point, np.float32]
    """
    initial_row, initial_col = initial_position
    new_rows = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float32) * precision + initial_row
    new_cols = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float32) * precision + initial_col
    candidates = filter_method(dichotomy_window, [new_rows, new_cols])
    # In case a NaN is present in the kernel, candidates will be all-NaNs. Let’s restore initial_position value so
    # that best candidate search will be able to find it.
    candidates[4] = initial_value
    best_index = cost_selection_method(candidates)
    return Point(new_rows[best_index], new_cols[best_index]), candidates[best_index]
