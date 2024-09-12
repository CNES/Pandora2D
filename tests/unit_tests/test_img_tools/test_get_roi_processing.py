# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Test get_roi_processing.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import pytest
import numpy as np

from pandora2d import img_tools


@pytest.fixture()
def default_roi():
    """
    Create a roi to test the get_roi_processing method
    """
    return {
        "col": {"first": 2, "last": 5},
        "row": {"first": 2, "last": 5},
        "margins": [2, 2, 2, 2],
    }


@pytest.mark.parametrize(
    ["col_disparity", "row_disparity", "expected"],
    [
        pytest.param(
            {"init": -30, "range": 30}, {"init": 1, "range": 1}, (60, 2, 2, 2), id="Negative disparity for columns"
        ),
        pytest.param(
            {"init": 1, "range": 1}, {"init": -30, "range": 30}, (2, 60, 2, 2), id="Negative disparity for rows"
        ),
        pytest.param(
            {"init": -30, "range": 30},
            {"init": -30, "range": 30},
            (60, 60, 2, 2),
            id="Negative disparity for columns and rows",
        ),
        pytest.param(
            {"init": 30, "range": 30},
            {"init": 30, "range": 30},
            (2, 2, 60, 60),
            id="Positive disparity for columns and rows",
        ),
        pytest.param(
            {"init": 0, "range": 1}, {"init": 0, "range": 1}, (2, 2, 2, 2), id="Margins greater than disparities"
        ),
        pytest.param(
            {"init": 0, "range": 3}, {"init": 0, "range": 3}, (3, 3, 3, 3), id="Margins lower than disparities"
        ),
    ],
)
def test_roi_with_negative_and_positive_disparities(default_roi, col_disparity, row_disparity, expected):
    """
    Test the get_roi_processing method with negative disparities
    """
    test_roi_column = img_tools.get_roi_processing(default_roi, col_disparity, row_disparity)

    assert test_roi_column["margins"] == expected
    assert test_roi_column == default_roi


@pytest.fixture
def positive_grid(left_img_shape, create_disparity_grid_fixture):
    """Create a positive disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width) with alternating rows of 6 and 8
    init_band = np.tile([[6], [8]], (height // 2 + 1, width))[:height, :]

    return create_disparity_grid_fixture(init_band, 2, "postive_disparity.tif")


@pytest.fixture
def negative_grid(left_img_shape, create_disparity_grid_fixture):
    """Create a negative disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width) with alternating rows of -5 and -7
    init_band = np.tile([[-5], [-7]], (height // 2 + 1, width))[:height, :]

    return create_disparity_grid_fixture(init_band, 2, "negative_disparity.tif")


@pytest.fixture
def lower_than_margins_grid(left_img_shape, create_disparity_grid_fixture):
    """
    Create a disparity grid with disparity lower than default_roi margins
    and save it in tmp
    """

    height, width = left_img_shape

    init_band = np.full((height, width), 0)

    return create_disparity_grid_fixture(init_band, 1, "lower_than_margins_disparity.tif")


@pytest.mark.parametrize(
    ["col_disparity", "row_disparity", "expected"],
    [
        pytest.param("second_correct_grid", "correct_grid", (26, 5, 10, 8), id="Negative and positive disparities"),
        pytest.param("negative_grid", "positive_grid", (9, 4, 3, 10), id="Negative disparities for columns"),
        pytest.param("positive_grid", "negative_grid", (4, 9, 10, 3), id="Negative disparities for rows"),
        pytest.param(
            "lower_than_margins_grid",
            "lower_than_margins_grid",
            (2, 2, 2, 2),
            id="Margins greater than disparities",
        ),
    ],
)
def test_roi_with_negative_and_positive_disparities_grids(default_roi, col_disparity, row_disparity, expected, request):
    """
    Test the get_roi_processing method with grid disparities
    """
    test_roi_column = img_tools.get_roi_processing(
        default_roi, request.getfixturevalue(col_disparity), request.getfixturevalue(row_disparity)
    )

    assert test_roi_column["margins"] == expected
    assert test_roi_column == default_roi
