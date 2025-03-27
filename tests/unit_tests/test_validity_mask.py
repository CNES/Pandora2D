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
#
"""
Test methods from criteria.py file linked to validity_mask creation
"""

# pylint: disable=too-few-public-methods
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import xarray as xr

from pandora2d import matching_cost, criteria
from pandora2d.img_tools import add_disparity_grid
from pandora2d.constants import Criteria


@pytest.fixture()
def make_image():
    """
    Create image dataset
    """

    def inner(row_disparity, col_disparity, mask_data):
        """Make image"""
        row, col = (6, 8)

        data = np.random.uniform(0, row * col, (row, col))

        return xr.Dataset(
            {
                "im": (["row", "col"], data),
                "msk": (["row", "col"], mask_data),
            },
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "invalid_disparity": np.nan,
            },
        ).pipe(add_disparity_grid, col_disparity, row_disparity)

    return inner


@pytest.fixture()
def make_cost_volumes(make_image, request):
    """
    Instantiate a matching_cost and compute cost_volumes
    """

    cfg = {
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": request.param["window_size"],
                "step": request.param["step"],
                "subpix": request.param["subpix"],
            }
        }
    }

    disp_row = request.param["row_disparity"]
    disp_col = request.param["col_disparity"]

    img_left = make_image(disp_row, disp_col, request.param["msk_left"])
    img_right = make_image(disp_row, disp_col, request.param["msk_right"])

    matching_cost_object = matching_cost.MatchingCostRegistry.get(
        cfg["pipeline"]["matching_cost"]["matching_cost_method"]
    )
    matching_cost_ = matching_cost_object(cfg["pipeline"]["matching_cost"])

    matching_cost_.allocate(img_left=img_left, img_right=img_right, cfg=cfg)

    cost_volumes = matching_cost_.compute_cost_volumes(img_left=img_left, img_right=img_right)

    return img_left, img_right, cost_volumes


class TestAllocateValidityDataset:
    """
    Test that the validity xr.Dataset is correctly allocated.
    """

    @pytest.mark.parametrize(
        ["make_cost_volumes"],
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
                id="Classic case",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.full((6, 8), 0),
                    "step": [1, 1],
                    "subpix": 2,
                    "window_size": 1,
                },
                id="Subpix=2",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.full((6, 8), 0),
                    "step": [1, 1],
                    "subpix": 4,
                    "window_size": 1,
                },
                id="Subpix=4",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_allocate_validity_dataset(self, make_cost_volumes):
        """
        Test the allocate_validity_dataset method
        """

        img_left, img_right, cost_volumes = make_cost_volumes

        criteria_dataarray = criteria.get_criteria_dataarray(
            left_image=img_left, right_image=img_right, cv=cost_volumes
        )

        allocated_validity_mask = criteria.allocate_validity_dataset(criteria_dataarray)

        assert allocated_validity_mask.sizes["row"] == criteria_dataarray.sizes["row"]
        assert allocated_validity_mask.sizes["col"] == criteria_dataarray.sizes["col"]
        # The dimension 'criteria' is the same size as the Enum Criteria
        # because there is a band for each criteria except the 'Valid' and a band for the global 'validity_mask'.
        assert allocated_validity_mask.sizes["criteria"] == len(Criteria.__members__)


class TestValidityMaskBand:
    """
    Test that the validity mask band obtained with the criteria dataarray is correct.
    """

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
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                id="No mask and window_size=1",
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
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 1, 1, 1, 1, 1, 1, 2],
                        [2, 1, 1, 0, 0, 1, 1, 2],
                        [2, 1, 1, 0, 0, 1, 1, 2],
                        [2, 1, 1, 1, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="No mask and window_size=3",
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
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 1, 1, 1, 1, 2, 2],
                        [2, 2, 1, 1, 1, 1, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="No mask and window_size=5",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 2, 0, 1],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                            ]
                        )
                    ),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    "subpix": 1,
                    "window_size": 1,
                },
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 0, 0, 2, 1, 1],
                        [1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 2, 0, 0, 0, 1, 1],
                        [1, 1, 0, 2, 0, 2, 1, 2],
                        [1, 1, 1, 1, 1, 2, 1, 1],
                    ]
                ),
                id="Left mask, window_size=1",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 2, 0, 1],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                            ]
                        )
                    ),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    "subpix": 1,
                    "window_size": 3,
                },
                np.array(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 1, 2, 1, 2],
                        [2, 2, 2, 2, 0, 1, 1, 2],
                        [2, 1, 2, 2, 2, 1, 2, 2],
                        [2, 1, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="Left mask, window_size=3",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    "subpix": 1,
                    "window_size": 1,
                },
                np.array(
                    (
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    )
                ),
                id="Right mask, window_size=1",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    # Temporarily, the criteria for subpixel disparities
                    # are raised by following a nearest neighbor strategy.
                    "subpix": 2,
                    "window_size": 1,
                },
                np.array(
                    (
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    )
                ),
                id="Right mask, window_size=1, subpix=2",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    "subpix": 1,
                    "window_size": 3,
                },
                np.array(
                    (
                        [
                            [2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2],
                        ]
                    )
                ),
                id="Right mask, window_size=3",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    # Temporarily, the criteria for subpixel disparities
                    # are raised by following a nearest neighbor strategy.
                    "subpix": 2,
                    "window_size": 3,
                },
                np.array(
                    (
                        [
                            [2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2],
                        ]
                    )
                ),
                id="Right mask, window_size=3, subpix=2",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.full((6, 8), 0),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    # Temporarily, the criteria for subpixel disparities
                    # are raised by following a nearest neighbor strategy.
                    "subpix": 4,
                    "window_size": 3,
                },
                np.array(
                    (
                        [
                            [2, 2, 2, 2, 2, 2, 2, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 1, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 1, 1, 1, 1, 2],
                            [2, 2, 2, 2, 2, 2, 2, 2],
                        ]
                    )
                ),
                id="Right mask, window_size=3, subpix=4",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 2, 0, 1],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                            ]
                        )
                    ),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    "subpix": 1,
                    "window_size": 3,
                },
                np.array(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 1, 2, 1, 2],
                        [2, 2, 2, 2, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 1, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="Left and right masks, window_size=3",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 2, 0, 1],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                            ]
                        )
                    ),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    # Temporarily, the criteria for subpixel disparities
                    # are raised by following a nearest neighbor strategy.
                    "subpix": 2,
                    "window_size": 3,
                },
                np.array(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 1, 2, 1, 2],
                        [2, 2, 2, 2, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 1, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="Left and right masks, window_size=3, subpix=2",
            ),
            pytest.param(
                {
                    "row_disparity": {"init": 0, "range": 1},
                    "col_disparity": {"init": 0, "range": 2},
                    "msk_left": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 2, 0, 1],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                            ]
                        )
                    ),
                    "msk_right": np.array(
                        (
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 2, 2, 2, 0],
                                [0, 2, 0, 0, 0, 0, 0, 0],
                                [0, 2, 2, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ]
                        )
                    ),
                    "step": [1, 1],
                    # Temporarily, the criteria for subpixel disparities
                    # are raised by following a nearest neighbor strategy.
                    "subpix": 4,
                    "window_size": 3,
                },
                np.array(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 1, 2, 1, 2],
                        [2, 2, 2, 2, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 1, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                ),
                id="Left and right masks, window_size=3, subpix=4",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_validity_mask(self, make_cost_volumes, expected):
        """
        Test that the produced validity mask bands are correct according to:
            - disparities
            - window size
            - left and right masks
        """

        img_left, img_right, cost_volumes = make_cost_volumes

        criteria_dataarray = criteria.get_criteria_dataarray(
            left_image=img_left, right_image=img_right, cv=cost_volumes
        )

        validity_mask_band = criteria.get_validity_mask_band(criteria_dataarray)

        np.testing.assert_array_equal(validity_mask_band, expected)


class TestLeftBorderBand:
    """
    Test that the left border (P2D_LEFT_BORDER) band obtained with the criteria dataarray is correct.
    """

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
    def test_left_border_band(self, make_cost_volumes, expected):
        """
        Test that the produced left border bands are correct according to the window size.
        """

        img_left, img_right, cost_volumes = make_cost_volumes

        criteria_dataarray = criteria.get_criteria_dataarray(
            left_image=img_left, right_image=img_right, cv=cost_volumes
        )

        left_border_band = (
            criteria.get_validity_dataset(criteria_dataarray).sel(criteria="P2D_LEFT_BORDER")["validity"].data
        )

        np.testing.assert_array_equal(left_border_band, expected)
