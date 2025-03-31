#!/usr/bin/env python
#
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
Tests subpix parameter in matching cost step
"""

import pytest
import numpy as np
import xarray as xr
from pandora2d.img_tools import add_disparity_grid
from pandora2d.state_machine import Pandora2DMachine


class TestSubpix:
    """
    Test subpix parameter in matching cost
    """

    @pytest.fixture()
    def create_datasets(self, data_left, data_right):
        """
        Creates left and right datasets to test subpix efficiency
        """

        left = xr.Dataset(
            {"im": (["row", "col"], data_left)},
            coords={"row": np.arange(data_left.shape[0]), "col": np.arange(data_left.shape[1])},
        )

        add_disparity_grid(left, {"init": 1, "range": 2}, {"init": 1, "range": 2})

        left.attrs.update(
            {
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": None,
            }
        )

        right = xr.Dataset(
            {"im": (["row", "col"], data_right)},
            coords={"row": np.arange(data_right.shape[0]), "col": np.arange(data_right.shape[1])},
        )
        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": None,
        }

        return left, right

    @pytest.fixture()
    def config(self, correct_input_for_functional_tests, subpix):
        """
        Creates configuration to test subpix efficiency
        """
        return {
            **correct_input_for_functional_tests,
            "pipeline": {
                "matching_cost": {"matching_cost_method": "ssd", "window_size": 1, "step": [1, 1], "subpix": subpix},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -5},
            },
        }

    @pytest.mark.parametrize(
        ["subpix", "data_left", "data_right", "disp_map", "gt_disp_map"],
        [
            pytest.param(
                1,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.5, 1.5, 1.5, 1.5, 1.5],
                        [2.5, 2.5, 2.5, 2.5, 2.5],
                        [3.5, 3.5, 3.5, 3.5, 3.5],
                        [4.5, 4.5, 4.5, 4.5, 4.5],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    (
                        [0.0, 0.0, 0.0, 0.0, 0.0],  # the subpixel shift is not identified in the disparity map
                        [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=1 and rows shifted by -0.5",
            ),
            pytest.param(
                2,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.5, 1.5, 1.5, 1.5, 1.5],
                        [2.5, 2.5, 2.5, 2.5, 2.5],
                        [3.5, 3.5, 3.5, 3.5, 3.5],
                        [4.5, 4.5, 4.5, 4.5, 4.5],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],  # shift < 0: first row is not shifted
                        [-0.5, -0.5, -0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5, -0.5, -0.5],
                    ]
                ),
                id="Subpix=2 and rows shifted by -0.5",
            ),
            pytest.param(
                2,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1.5, 1.5, 1.5, 1.5, 1.5],
                        [2.5, 2.5, 2.5, 2.5, 2.5],
                        [3.5, 3.5, 3.5, 3.5, 3.5],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0, 0]]
                ),  # shift >  0: last row is not shifted
                id="Subpix=2 and rows shifted by 0.5",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.25, 1.25, 1.25, 1.25, 1.25],
                        [2.25, 2.25, 2.25, 2.25, 2.25],
                        [3.25, 3.25, 3.25, 3.25, 3.25],
                        [4.25, 4.25, 4.25, 4.25, 4.25],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],  # shift < 0: first row is not shifted
                        [-0.25, -0.25, -0.25, -0.25, -0.25],
                        [-0.25, -0.25, -0.25, -0.25, -0.25],
                        [-0.25, -0.25, -0.25, -0.25, -0.25],
                    ]
                ),
                id="Subpix=4 and rows shifted by -0.25",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.75, 0.75, 0.75, 0.75, 0.75],
                        [1.75, 1.75, 1.75, 1.75, 1.75],
                        [2.75, 2.75, 2.75, 2.75, 2.75],
                        [3.75, 3.75, 3.75, 3.75, 3.75],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [
                        [0.25, 0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25, 0.25],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),  # shift >  0: last row is not shifted
                id="Subpix=4 and rows shifted by 0.25",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.75, 1.75, 1.75, 1.75, 1.75],
                        [2.75, 2.75, 2.75, 2.75, 2.75],
                        [3.75, 3.75, 3.75, 3.75, 3.75],
                        [4.75, 4.75, 4.75, 4.75, 4.75],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],  # shift < 0: first row is not shifted
                        [-0.75, -0.75, -0.75, -0.75, -0.75],
                        [-0.75, -0.75, -0.75, -0.75, -0.75],
                        [-0.75, -0.75, -0.75, -0.75, -0.75],
                    ]
                ),
                id="Subpix=4 and rows shifted by -0.75",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.25, 0.25, 0.25, 0.25, 0.25],
                        [1.25, 1.25, 1.25, 1.25, 1.25],
                        [2.25, 2.25, 2.25, 2.25, 2.25],
                        [3.25, 3.25, 3.25, 3.25, 3.25],
                    ),
                    dtype=np.float64,
                ),
                "row_map",
                np.array(
                    [
                        [0.75, 0.75, 0.75, 0.75, 0.75],
                        [0.75, 0.75, 0.75, 0.75, 0.75],
                        [0.75, 0.75, 0.75, 0.75, 0.75],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),  # shift >  0: last row is not shifted
                id="Subpix=4 and rows shifted by 0.75",
            ),
            pytest.param(
                1,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.0, -1.0, -1.0, -1.0, -1.0],  # the subpixel shift is not identified in the disparity map
                        [0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.0, -1.0, -1.0, -1.0, -1.0],
                        [0.0, -1.0, -1.0, -1.0, -1.0],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=1 and columns shifted by -0.5",
            ),
            pytest.param(
                2,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.0, -0.5, -0.5, -0.5, -0.5],  # shift < 0: first column is not shifted
                        [0.0, -0.5, -0.5, -0.5, -0.5],
                        [0.0, -0.5, -0.5, -0.5, -0.5],
                        [0.0, -0.5, -0.5, -0.5, -0.5],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=2 and columns shifted by -0.5",
            ),
            pytest.param(
                2,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.5, 1.5, 2.5, 3.5, 4.5],
                        [0.5, 1.5, 2.5, 3.5, 4.5],
                        [0.5, 1.5, 2.5, 3.5, 4.5],
                        [0.5, 1.5, 2.5, 3.5, 4.5],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.5, 0.5, 0.5, 0.5, 0.0],  # shift > 0: last column is not shifted
                        [0.5, 0.5, 0.5, 0.5, 0.0],
                        [0.5, 0.5, 0.5, 0.5, 0.0],
                        [0.5, 0.5, 0.5, 0.5, 0.0],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=2 and columns shifted by 0.5",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.25, 2.25, 3.25, 4.25, 5.25],
                        [1.25, 2.25, 3.25, 4.25, 5.25],
                        [1.25, 2.25, 3.25, 4.25, 5.25],
                        [1.25, 2.25, 3.25, 4.25, 5.25],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.0, -0.25, -0.25, -0.25, -0.25],  # shift < 0: first column is not shifted
                        [0.0, -0.25, -0.25, -0.25, -0.25],
                        [0.0, -0.25, -0.25, -0.25, -0.25],
                        [0.0, -0.25, -0.25, -0.25, -0.25],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=4 and columns shifted by -0.25",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.75, 1.75, 2.75, 3.75, 4.75],
                        [0.75, 1.75, 2.75, 3.75, 4.75],
                        [0.75, 1.75, 2.75, 3.75, 4.75],
                        [0.75, 1.75, 2.75, 3.75, 4.75],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.25, 0.25, 0.25, 0.25, 0.0],  # shift > 0: last column is not shifted
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=2 and columns shifted by 0.25",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [1.75, 2.75, 3.75, 4.75, 5.75],
                        [1.75, 2.75, 3.75, 4.75, 5.75],
                        [1.75, 2.75, 3.75, 4.75, 5.75],
                        [1.75, 2.75, 3.75, 4.75, 5.75],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.0, -0.75, -0.75, -0.75, -0.75],  # shift < 0: first column is not shifted
                        [0.0, -0.75, -0.75, -0.75, -0.75],
                        [0.0, -0.75, -0.75, -0.75, -0.75],
                        [0.0, -0.75, -0.75, -0.75, -0.75],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=4 and columns shifted by -0.75",
            ),
            pytest.param(
                4,
                np.array(
                    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                    dtype=np.float64,
                ),
                np.array(
                    (
                        [0.25, 1.25, 2.25, 3.25, 4.25],
                        [0.25, 1.25, 2.25, 3.25, 4.25],
                        [0.25, 1.25, 2.25, 3.25, 4.25],
                        [0.25, 1.25, 2.25, 3.25, 4.25],
                    ),
                    dtype=np.float64,
                ),
                "col_map",
                np.array(
                    (
                        [0.75, 0.75, 0.75, 0.75, 0.0],  # shift > 0: last column is not shifted
                        [0.75, 0.75, 0.75, 0.75, 0.0],
                        [0.75, 0.75, 0.75, 0.75, 0.0],
                        [0.75, 0.75, 0.75, 0.75, 0.0],
                    ),
                    dtype=np.float64,
                ),
                id="Subpix=2 and columns shifted by 0.75",
            ),
        ],
    )
    def test_subpix(self, config, create_datasets, disp_map, gt_disp_map):
        """
        Tests that the subpix parameter in matching cost step produces
        valid disparity maps when there is a subpixel shift.
        """

        img_left, img_right = create_datasets

        pandora2d_machine = Pandora2DMachine()

        pandora2d_machine.run_prepare(img_left, img_right, config)
        pandora2d_machine.run("matching_cost", config)
        pandora2d_machine.run("disparity", config)

        np.testing.assert_array_equal(pandora2d_machine.dataset_disp_maps[disp_map], gt_disp_map)
