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
Test remove_roi_margins.
"""

import pytest
import numpy as np
import xarray as xr

from pandora2d.img_tools import remove_roi_margins


def create_dataset(row, col):
    """
    Create dataset to test remove_roi_margins method
    """

    data = np.full((len(row), len(col)), 1)
    data_validity = np.full((len(row), len(col), 2), 0)

    criteria_values = ["validity_mask", "criteria_1"]
    coords = {"row": row, "col": col}
    dims = ("row", "col")

    dataset = xr.Dataset(
        {
            "row_map": xr.DataArray(data, dims=dims, coords=coords),
            "col_map": xr.DataArray(data, dims=dims, coords=coords),
            "correlation_score": xr.DataArray(data, dims=dims, coords=coords),
            "validity": xr.DataArray(
                data_validity, dims=("row", "col", "criteria"), coords={**coords, "criteria": criteria_values}
            ),
        },
    )

    return dataset


def make_cfg(roi, step):
    """
    Create user configuration to test remove_roi_margins method
    """
    return {
        "ROI": roi,
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 7, "step": step},
        },
    }


class TestRemoveRoiMargins:
    """
    Test remove_roi_margins function
    """

    # pylint:disable=too-few-public-methods

    @pytest.mark.parametrize(
        ["roi", "step", "row", "col", "row_gt", "col_gt"],
        [
            pytest.param(
                {
                    "col": {"first": 10, "last": 100},
                    "row": {"first": 10, "last": 100},
                    "margins": (3, 3, 3, 3),
                },
                [1, 1],
                np.arange(7, 104),
                np.arange(7, 104),
                np.arange(10, 101),
                np.arange(10, 101),
                id="Centered ROI, step_row=1 and step_col=1 ",
            ),
            pytest.param(
                {
                    "col": {"first": 10, "last": 100},
                    "row": {"first": 0, "last": 100},
                    "margins": (2, 2, 2, 2),
                },
                [1, 1],
                np.arange(0, 103),
                np.arange(8, 103),
                np.arange(0, 101),
                np.arange(10, 101),
                id="ROI overlap at the top, step_row=1 and step_col=1 ",
            ),
            pytest.param(
                {
                    "col": {"first": 10, "last": 100},
                    "row": {"first": 100, "last": 201},
                    "margins": (3, 3, 3, 3),
                },
                [1, 1],
                np.arange(97, 200),
                np.arange(7, 104),
                np.arange(100, 200),
                np.arange(10, 101),
                id="ROI overlap at the bottom, step_row=1 and step_col=1 ",
            ),
            pytest.param(
                {
                    "col": {"first": 0, "last": 100},
                    "row": {"first": 10, "last": 100},
                    "margins": (3, 3, 3, 3),
                },
                [1, 1],
                np.arange(7, 104),
                np.arange(0, 104),
                np.arange(10, 101),
                np.arange(0, 101),
                id="ROI overlap on the left, step_row=1 and step_col=1 ",
            ),
            pytest.param(
                {
                    "col": {"first": 100, "last": 202},
                    "row": {"first": 10, "last": 100},
                    "margins": (3, 3, 3, 3),
                },
                [1, 1],
                np.arange(7, 104),
                np.arange(97, 200),
                np.arange(10, 101),
                np.arange(100, 200),
                id="ROI overlap on the right, step_row=1 and step_col=1 ",
            ),
            pytest.param(
                {
                    "col": {"first": 110, "last": 200},
                    "row": {"first": 110, "last": 200},
                    "margins": (3, 3, 3, 3),
                },
                [2, 3],
                np.arange(108, 204, 2),  # step=2 --> we start at 108 even if margin=3
                np.arange(107, 204, 3),
                np.arange(110, 201, 2),
                np.arange(110, 201, 3),
                id="Centered ROI, step_row=2 and step_col=3 ",
            ),
            pytest.param(
                {
                    "col": {"first": 0, "last": 200},
                    "row": {"first": 110, "last": 200},
                    "margins": (3, 3, 3, 3),
                },
                [2, 3],
                np.arange(108, 204, 2),  # step=2 --> we start at 108 even if margin=3
                np.arange(0, 204, 3),
                np.arange(110, 201, 2),
                np.arange(0, 201, 3),
                id="ROI overlap on the left, step_row=2 and step_col=3 ",
            ),
            pytest.param(
                {
                    "col": {"first": 100, "last": 203},
                    "row": {"first": 10, "last": 100},
                    "margins": (3, 3, 3, 3),
                },
                [3, 2],
                np.arange(7, 104, 3),
                np.arange(98, 200, 2),  # step=2 --> we start at 98 even if margin=3
                np.arange(10, 101, 3),
                np.arange(100, 200, 2),
                id="ROI overlap on the right, step_row=3 and step_col=2 ",
            ),
            pytest.param(
                {
                    "col": {"first": 10, "last": 100},
                    "row": {"first": 0, "last": 100},
                    "margins": (3, 3, 3, 3),
                },
                [4, 2],
                np.arange(0, 104, 4),
                np.arange(8, 104, 2),  # step=2 --> we start at 8 even if margin=3
                np.arange(0, 101, 4),
                np.arange(10, 101, 2),
                id="ROI overlap at the top, step_row=4 and step_col=2",
            ),
            pytest.param(
                {
                    "col": {"first": 10, "last": 100},
                    "row": {"first": 100, "last": 203},
                    "margins": (3, 3, 3, 3),
                },
                [5, 3],
                np.arange(100, 200, 5),  # step=5 --> we start at 100 even if margin=3
                np.arange(7, 104, 3),
                np.arange(100, 200, 5),
                np.arange(10, 101, 3),
                id="ROI overlap at the bottom, step_row=5 and step_col=3",
            ),
        ],
    )
    def test_remove_roi_margins(self, roi, step, row, col, row_gt, col_gt):
        """
        Test remove_roi_margins method
        """

        # Create user configuration with given roi and step
        cfg = make_cfg(roi, step)

        # Create dataset input for remove_roi_margins
        # row & col are supposed to be the correct coordinates according to the roi and the step
        dataset = create_dataset(row, col)

        # Create ground truth dataset
        dataset_gt = create_dataset(row_gt, col_gt)

        dataset_test = remove_roi_margins(dataset, cfg)

        xr.testing.assert_identical(dataset_test, dataset_gt)
