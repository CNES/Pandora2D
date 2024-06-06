#!/usr/bin/env python
#
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
Tests disparity margins in matching cost step
"""

import pytest
import numpy as np
import xarray as xr

from pandora2d.img_tools import add_disparity_grid
from pandora2d.state_machine import Pandora2DMachine


class TestDisparityMargins:
    """
    Test disparity margins in the cost volume
    """

    @pytest.fixture()
    def create_datasets(self):
        """
        Creates left and right datasets
        """

        data = np.full((10, 10), 1)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        add_disparity_grid(left, [1, 3], [-2, 2])

        left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "col_disparity_source": [1, 3],
            "row_disparity_source": [-2, 2],
        }

        data = np.full((10, 10), 1)
        right = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "col_disparity_source": [1, 3],
            "row_disparity_source": [-2, 2],
        }

        return left, right

    @pytest.fixture()
    def config(self, subpix, refinement_config, matching_cost_method):
        return {
            # "input": input_config,
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": matching_cost_method,
                    "window_size": 3,
                    "step": [1, 1],
                    "subpix": subpix,
                },
                "disparity": {"disparity_method": "wta", "invalid_disparity": -6},
                "refinement": refinement_config,
            }
        }

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc"])
    @pytest.mark.parametrize(
        ["subpix", "refinement_config", "cv_shape_expected", "disp_col_expected", "disp_row_expected"],
        [
            pytest.param(
                1,
                {"refinement_method": "dichotomy", "iterations": 1, "filter": "bicubic"},
                (10, 10, 6, 8),
                [0, 1, 2, 3, 4, 5],
                [-3, -2, -1, 0, 1, 2, 3, 4],
                id="Subpix=1 and refinement_method=dichotomy",
            ),
            pytest.param(
                1,
                {
                    "refinement_method": "interpolation",
                },
                (10, 10, 9, 11),
                [-2, -1, 0, 1, 2, 3, 4, 5, 6],
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                id="Subpix=1 and refinement_method=interpolation",
            ),
            pytest.param(
                1,
                {
                    "refinement_method": "optical_flow",
                },
                (10, 10, 5, 7),
                [0, 1, 2, 3, 4],
                [-3, -2, -1, 0, 1, 2, 3],
                id="Subpix=1 and refinement_method=optical_flow",
            ),
            pytest.param(
                2,
                {"refinement_method": "dichotomy", "iterations": 1, "filter": "bicubic"},
                (10, 10, 11, 15),
                [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                id="Subpix=2 and refinement_method=dichotomy",
            ),
            pytest.param(
                2,
                {
                    "refinement_method": "interpolation",
                },
                (10, 10, 17, 21),
                [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
                [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                id="Subpix=2 and refinement_method=interpolation",
            ),
            pytest.param(
                2,
                {
                    "refinement_method": "optical_flow",
                },
                (10, 10, 9, 13),
                [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                id="Subpix=2 and refinement_method=optical_flow",
            ),
            pytest.param(
                4,
                {"refinement_method": "dichotomy", "iterations": 1, "filter": "bicubic"},
                (10, 10, 21, 29),
                np.arange(0, 5.25, 0.25),
                np.arange(-3, 4.25, 0.25),
                id="Subpix=4 and refinement_method=dichotomy",
            ),
            pytest.param(
                4,
                {
                    "refinement_method": "interpolation",
                },
                (10, 10, 33, 41),
                np.arange(-2, 6.25, 0.25),
                np.arange(-5, 5.25, 0.25),
                id="Subpix=4 and refinement_method=interpolation",
            ),
            pytest.param(
                4,
                {
                    "refinement_method": "optical_flow",
                },
                (10, 10, 17, 25),
                [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
                np.arange(-3, 3.25, 0.25),
                id="Subpix=4 and refinement_method=optical_flow",
            ),
        ],
    )
    def test_disparity_margins_in_cost_volumes(
        self, cv_shape_expected, disp_col_expected, disp_row_expected, config, create_datasets
    ):
        """
        Test that the disparity margins are correctly added in the cost volumes
        according to the refinement margins.
        """

        pandora2d_machine = Pandora2DMachine()

        img_left, img_right = create_datasets

        pandora2d_machine.check_conf(config)
        pandora2d_machine.run_prepare(img_left, img_right, config)
        pandora2d_machine.run("matching_cost", config)

        cost_volumes = pandora2d_machine.cost_volumes["cost_volumes"]

        np.testing.assert_array_equal(cost_volumes.shape, cv_shape_expected)
        np.testing.assert_array_equal(cost_volumes.disp_col, disp_col_expected)
        np.testing.assert_array_equal(cost_volumes.disp_row, disp_row_expected)
