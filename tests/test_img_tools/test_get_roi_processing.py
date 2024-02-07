# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
        pytest.param([-60, 0], [0, 2], [60, 2, 2, 2], id="Negative disparitie for columns"),
        pytest.param([0, 2], [-60, 0], [2, 60, 2, 2], id="Negative disparitie for rows"),
        pytest.param([-60, 0], [-60, 0], [60, 60, 2, 2], id="Negative disparitie for columns and rows"),
        pytest.param([0, 60], [0, 60], [2, 2, 60, 60], id="Negative disparitie for columns and rows"),
    ],
)
def test_roi_with_negative_and_positive_disparities(default_roi, col_disparity, row_disparity, expected):
    """
    Test the get_roi_processing method with negative disparities
    """
    test_roi_column = img_tools.get_roi_processing(default_roi, col_disparity, row_disparity)
    default_roi["margins"] = expected

    assert test_roi_column == default_roi
