#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Test that the left border (P2D_LEFT_BORDER) band obtained with the criteria dataarray is correct.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def criteria_name():
    return Criteria.P2D_LEFT_BORDER.name


@pytest.mark.parametrize(
    ["make_cost_volumes", "expected"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            np.full((6, 8), 0),
            id="Window_size=1",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="Window_size=3",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 5,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="Window_size=5",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 7,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="Window_size=7",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_window_size(make_cost_volumes, criteria_name, expected):
    """
    Test that the produced left border bands are correct according to the window size.
    """

    cost_volumes = make_cost_volumes

    criteria_dataarray = cost_volumes.criteria

    result = criteria.get_validity_dataset(criteria_dataarray).sel(criteria=criteria_name)["validity"].data

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["make_cost_volumes", "expected"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 2,
                "window_size": 5,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="Window_size=5, subpix=2",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 4,
                "window_size": 5,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="Window_size=5, subpix=4",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_subpix(make_cost_volumes, criteria_name, expected):
    """
    Test that the produced left border bands are correct according to the subpix.
    """

    cost_volumes = make_cost_volumes

    criteria_dataarray = cost_volumes.criteria

    result = criteria.get_validity_dataset(criteria_dataarray).sel(criteria=criteria_name)["validity"].data

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["make_cost_volumes", "expected"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [2, 3],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [1, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ),
            id="Window_size=3 and step=[2,3]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 2],
                "subpix": 1,
                "window_size": 5,
            },
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
            id="Window_size=5 and step=[1,2]",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_step(make_cost_volumes, criteria_name, expected):
    """
    Test that the produced left border bands are correct according to the step.
    """

    cost_volumes = make_cost_volumes

    criteria_dataarray = cost_volumes.criteria

    result = criteria.get_validity_dataset(criteria_dataarray).sel(criteria=criteria_name)["validity"].data

    np.testing.assert_array_equal(result, expected)
