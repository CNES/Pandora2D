#!/usr/bin/env python
# coding: utf8
#
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
#

"""
Test Matching cost class
"""
# pylint: disable=redefined-outer-name
from typing import NamedTuple

import numpy as np
import xarray as xr
import pytest
from rasterio import Affine
import json_checker

from pandora2d import matching_cost


@pytest.fixture()
def create_image():
    """Create an image."""

    def create(data):
        return xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            },
        )

    return create


@pytest.fixture()
def left_zncc(create_image):
    """Left image for Znnc."""
    data = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [3, 4, 5, 6, 7],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    return create_image(data)


@pytest.fixture()
def right_zncc(create_image):
    """Right image for Znnc."""
    data = np.array(
        (
            [
                [1, 1, 1, 1, 1],
                [3, 4, 5, 6, 7],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ),
        dtype=np.float64,
    )
    return create_image(data)


@pytest.fixture()
def null_disparity_grid():
    return np.zeros((3, 3))


@pytest.fixture()
def positive_disparity_grid():
    return np.full((3, 3), 1)


@pytest.fixture()
def negative_disparity_grid():
    return np.full((3, 3), -1)


class DisparityGrids(NamedTuple):
    """
    NamedTuple used to group disparity grids together in tests.
    """

    col_min: np.ndarray
    col_max: np.ndarray
    row_min: np.ndarray
    row_max: np.ndarray


class StepData(NamedTuple):
    """NamedTuple used to group data related to Step together."""

    left: xr.Dataset
    right: xr.Dataset
    full_matching_cost: np.ndarray
    disparity_grids: DisparityGrids


@pytest.fixture()
def data_with_null_disparity(left_zncc, right_zncc, null_disparity_grid):
    """
    Coherent Data for test_step.
    """
    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [[[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]]],
            [[[np.nan]], [[-0.45]], [[-0.47058824]], [[-0.48076922]], [[np.nan]]],
            [[[np.nan]], [[-0.45]], [[-0.47058824]], [[-0.48076922]], [[np.nan]]],
            [[[np.nan]], [[0.0]], [[0.0]], [[0.0]], [[np.nan]]],
            [[[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]]],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [0, 0]
    left_zncc.attrs["row_disparity_source"] = [0, 0]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_positive_disparity_in_col(left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid):
    """Coherent Data for test_step."""
    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=positive_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan]],
                [[-0.45], [-0.460179]],
                [[-0.47058824], [-0.4756515]],
                [[-0.48076922], [np.nan]],
                [[np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan]],
                [[-0.45], [-0.460179]],
                [[-0.47058824], [-0.4756515]],
                [[-0.48076922], [np.nan]],
                [[np.nan], [np.nan]],
            ],
            [[[np.nan], [np.nan]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [np.nan]], [[np.nan], [np.nan]]],
            [
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [0, 1]
    left_zncc.attrs["row_disparity_source"] = [0, 0]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_positive_disparity_in_row(left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid):
    """Coherent Data for test_step."""
    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=positive_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [[[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]]],
            [
                [[np.nan, np.nan]],
                [[-0.45, -0.45]],
                [[-0.47058824, -0.47058824]],
                [[-0.48076922, -0.48076922]],
                [[np.nan, np.nan]],
            ],
            [[[np.nan, np.nan]], [[-0.45, 0.0]], [[-0.47058824, 0.0]], [[-0.48076922, 0.0]], [[np.nan, np.nan]]],
            [[[np.nan, np.nan]], [[0.0, np.nan]], [[0.0, np.nan]], [[0.0, np.nan]], [[np.nan, np.nan]]],
            [[[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]]],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [0, 0]
    left_zncc.attrs["row_disparity_source"] = [0, 1]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_negative_disparity_in_col(left_zncc, right_zncc, null_disparity_grid, negative_disparity_grid):
    """Coherent Data for test_step."""
    disparity_grids = DisparityGrids(
        col_min=negative_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan]],
                [[np.nan], [-0.45]],
                [[-0.460179], [-0.47058824]],
                [[-0.4756515], [-0.48076922]],
                [[np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan]],
                [[np.nan], [-0.45]],
                [[-0.460179], [-0.47058824]],
                [[-0.4756515], [-0.48076922]],
                [[np.nan], [np.nan]],
            ],
            [[[np.nan], [np.nan]], [[np.nan], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[np.nan], [np.nan]]],
            [
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
                [[np.nan], [np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [-1, 0]
    left_zncc.attrs["row_disparity_source"] = [0, 0]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_negative_disparity_in_row(left_zncc, right_zncc, null_disparity_grid, negative_disparity_grid):
    """Coherent Data for test_step."""
    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=negative_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [[[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]]],
            [
                [[np.nan, np.nan]],
                [[np.nan, -0.45]],
                [[np.nan, -0.47058824]],
                [[np.nan, -0.48076922]],
                [[np.nan, np.nan]],
            ],
            [[[np.nan, np.nan]], [[1.0, -0.45]], [[1.0, -0.47058824]], [[1.0, -0.48076922]], [[np.nan, np.nan]]],
            [[[np.nan, np.nan]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[np.nan, np.nan]]],
            [[[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]], [[np.nan, np.nan]]],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [0, 0]
    left_zncc.attrs["row_disparity_source"] = [-1, 0]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_disparity_negative_in_row_and_positive_in_col(
    left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid, negative_disparity_grid
):
    """Coherent Data for test_step."""
    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=positive_disparity_grid,
        row_min=negative_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, -0.45], [np.nan, -0.460179]],
                [[np.nan, -0.47058824], [np.nan, -0.4756515]],
                [[np.nan, -0.48076922], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[1.0, -0.45], [0.99705446, -0.460179]],
                [[1.0, -0.47058824], [0.99886817, -0.4756515]],
                [[1.0, -0.48076922], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[1.0, 0.0], [0.99705446, 0.0]],
                [[1.0, 0.0], [0.99886817, 0.0]],
                [[1.0, 0.0], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    left_zncc.attrs["col_disparity_source"] = [0, 1]
    left_zncc.attrs["row_disparity_source"] = [-1, 0]
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


def test_step_configuration():
    """
    Test step in matching_cost configuration
    """

    matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3]})

    # Test with a negative step : test should fail
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [-2, 3]})

    # Test with a one size list step : test should fail
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2]})

    # Test with a three elements list step : test should fail
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3, 4]})

    # Test with a str elements list step : test should fail
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": ["2", 3]})


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "data_with_null_disparity",
        "data_with_positive_disparity_in_col",
        "data_with_positive_disparity_in_row",
        "data_with_negative_disparity_in_col",
        "data_with_negative_disparity_in_row",
        "data_with_disparity_negative_in_row_and_positive_in_col",
    ],
)
@pytest.mark.parametrize("col_step", [1, 2, pytest.param(5, id="Step gt image")])
@pytest.mark.parametrize("row_step", [1, 2, pytest.param(5, id="Step gt image")])
def test_steps(request, data_fixture_name, col_step, row_step):
    """We expect step to work."""
    data = request.getfixturevalue(data_fixture_name)

    # sum of squared difference images self.left, self.right, window_size=3
    cfg = {"matching_cost_method": "zncc", "window_size": 3, "step": [row_step, col_step]}
    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg)
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=data.left,
        img_right=data.right,
        grid_min_col=data.disparity_grids.col_min,
        grid_max_col=data.disparity_grids.col_max,
        cfg=cfg,
    )
    # compute cost volumes
    zncc = matching_cost_matcher.compute_cost_volumes(
        img_left=data.left,
        img_right=data.right,
        grid_min_col=data.disparity_grids.col_min,
        grid_max_col=data.disparity_grids.col_max,
        grid_min_row=data.disparity_grids.row_min,
        grid_max_row=data.disparity_grids.row_max,
    )

    # indexes are : row, col, disp_x, disp_y
    np.testing.assert_equal(zncc["cost_volumes"].data, data.full_matching_cost[::row_step, ::col_step, :, :])
