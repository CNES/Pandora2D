# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
Module for Dichotomy refinement method (python version).
"""

import logging
from typing import Dict, Tuple, Union, NamedTuple, Callable

import numpy as np
import xarray as xr

from json_checker import And

from pandora2d.interpolation_filter import AbstractFilter
from . import refinement

COST_SELECTION_METHOD_MAPPING = {"min": np.nanargmin, "max": np.nanargmax}


@refinement.AbstractRefinement.register_subclass("dichotomy_python")
class DichotomyPython(refinement.AbstractRefinement):
    """Subpixel refinement method by dichotomy (python version)."""

    NB_MAX_ITER = 9
    schema = {
        "refinement_method": And(str, lambda x: x in ["dichotomy_python"]),
        "iterations": And(int, lambda it: it > 0),
        "filter": And(dict, lambda x: x["method"] in AbstractFilter.interpolation_filter_methods_avail),
    }

    def __init__(self, cfg: dict = None, _: list = None, __: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """

        super().__init__(cfg)
        fractional_shift = 2 ** -self.cfg["iterations"]
        self.filter = AbstractFilter(  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated
            self.cfg["filter"], fractional_shift=fractional_shift
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

        return self.filter.margins

    def refinement_method(  # pylint: disable=too-many-locals
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

        # Get disparities grid

        # Select correct rows and columns in case of a step different from 1.
        row_cv = cost_volumes.row.values
        col_cv = cost_volumes.col.values

        # Column's min, max disparities
        disp_min_col = img_left["col_disparity"].sel(band_disp="min", row=row_cv, col=col_cv).data
        disp_max_col = img_left["col_disparity"].sel(band_disp="max", row=row_cv, col=col_cv).data
        # Row's min, max disparities
        disp_min_row = img_left["row_disparity"].sel(band_disp="min", row=row_cv, col=col_cv).data
        disp_max_row = img_left["row_disparity"].sel(band_disp="max", row=row_cv, col=col_cv).data

        # start iterations after subpixel precision: `subpixel.bit_length() - 1` found which power of 2 subpixel is,
        # and we add 1 to start at next iteration
        first_iteration = cost_volumes.attrs["subpixel"].bit_length()
        precisions = [1 / 2**it for it in range(first_iteration, self.cfg["iterations"] + 1)]
        if first_iteration >= 0:
            logging.info(
                "With subpixel of `%s` the `%s` first dichotomy iterations will be skipped.",
                cost_volumes.attrs["subpixel"],
                first_iteration - 1,
            )

        # Convert disparity maps to np.array to optimise performance
        row_map = row_map.to_numpy()
        col_map = col_map.to_numpy()

        # See usage of np.nditer:
        # https://numpy.org/doc/stable/reference/arrays.nditer.html#modifying-array-values
        with np.nditer(
            [cost_values, row_map, col_map, disp_min_row, disp_max_row, disp_min_col, disp_max_col],
            op_flags=[
                ["readwrite"],
                ["readwrite"],
                ["readwrite"],
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["readonly"],
            ],
        ) as iterators:
            for cost_surface, (
                cost_value,
                disp_row_init,
                disp_col_init,
                d_row_min,
                d_row_max,
                d_col_min,
                d_col_max,
            ) in zip(cost_surfaces, iterators):

                # Invalid value
                if np.isnan(cost_value):
                    continue

                # If the best candidate found at the disparity step is at the edge of the row disparity range
                # we do no enter the dichotomy loop
                if disp_row_init in (d_row_min, d_row_max):
                    continue

                # If the best candidate found at the disparity step is at the edge of the col disparity range
                # we do no enter the dichotomy loop
                if disp_col_init in (d_col_min, d_col_max):
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
                            cost_surface,
                            precision,
                            (disp_row_init, disp_col_init),  # type: ignore # Reason: is 0 dim array
                            (pos_disp_row_init, pos_disp_col_init),
                            cost_value,  # type: ignore # Reason: is 0 dim array
                            self.filter,
                            cost_selection_method,
                        )
                    )

        logging.info(
            "Dichotomy precision reached: %s", precisions[-1] if precisions else 1 / 2 ** (first_iteration - 1)
        )

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
        self.cost_volumes["cost_volumes"].attrs.update({"subpixel": cost_volumes.attrs["subpixel"]})

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
    cost_surface: xr.DataArray,
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
    :type cost_surface: xr.Dataarray with subpix attribute
    :param precision: subpixellic disparity precision to use
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

    # Whatever the cost_surface.attrs["subpixel"] value, the first precision in the cost surface is always 0.5
    # Then we multiply by cost_surface.attrs["subpixel"] to get right new_cols and new_rows

    # When there is no subpixel (it equals to 1), precision shift and index shift match:
    # the precision shift between two points is 1. So shifting from 0.5 precision corresponds to shift index of 0.5.
    # But when there is a subpixel, they do not match anymore:
    # in this case, the precision shift between two points is 1/subpix.
    # So to get the index corresponding to a given precision shift, we need to multiply this value by subpix.
    # For example when subix equals 2, the precision shift between two points is 0.5 while the index shift is still 1.
    # So in this case, shifting from 0.5 precision corresponds to shift index of 1
    # (`index_shift = 1 = 0.5 * 2 = precision_shift * subpix`)
    # In the same way, shifting from 0.25 precision corresponds to shift index of 0.5
    # (`index_shift = 0.5 = 0.25 * 2 = precision_shift * subpix`)

    new_cols = disp_col_shifts * cost_surface.attrs["subpixel"] + initial_pos_disp_col
    new_rows = disp_row_shifts * cost_surface.attrs["subpixel"] + initial_pos_disp_row

    # New subpixel disparity values
    new_rows_disp = disp_row_shifts + initial_disp_row
    new_cols_disp = disp_col_shifts + initial_disp_col

    # Interpolate points at positions (new_rows[i], new_cols[i])
    candidates = filter_dicho.interpolate(cost_surface.data, (new_cols, new_rows))

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
        Point(new_rows[best_index], new_cols[best_index]),
        new_rows_disp[best_index],
        new_cols_disp[best_index],
        candidates[best_index],
    )
