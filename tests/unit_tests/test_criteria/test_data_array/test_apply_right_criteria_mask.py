#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
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
Test apply_right_criteria_mask function
"""

import numpy as np
import pytest
import xarray as xr

from pandora2d import criteria
from pandora2d.constants import Criteria


class TestApplyRightCriteriaMask:
    """Test apply_right_criteria_mask function."""

    @pytest.fixture()
    def start_point_image(self):
        return [0, 0]

    @pytest.fixture()
    def mask_criteria_right(self, image, start_point_image):
        """Create a right mask with P2D_INVALID_MASK_RIGHT"""
        mask_criteria_right = xr.DataArray(
            np.full_like(image["msk"], Criteria.VALID, dtype=np.uint8),
            dims=["row", "col"],
            coords={
                "row": np.arange(start_point_image[0], start_point_image[0] + image["msk"].shape[0]),
                "col": np.arange(start_point_image[1], start_point_image[1] + image["msk"].shape[1]),
            },
        )

        invalid_right_mask = (image.msk != image.attrs["no_data_mask"]) & (image.msk != image.attrs["valid_pixels"])

        # Adding a criterion different from Criteria.VALID
        mask_criteria_right.data[invalid_right_mask.data] |= np.uint8(Criteria.P2D_INVALID_MASK_RIGHT)
        return mask_criteria_right

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize("img_size", [(4, 5)])
    @pytest.mark.parametrize(
        ["msk", "expected_criteria", "disp_col", "disp_row"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -2,
                -1,
                id="simple case with one P2D_INVALID_MASK_RIGHT",
            )
        ],
    )
    def test_simple_case(self, criteria_dataarray, mask_criteria_right, expected_criteria, disp_col, disp_row):
        """
        Test apply_right_criteria_mask with a simple case (without step or ROI)
        """
        criteria.apply_right_criteria_mask(criteria_dataarray, mask_criteria_right)

        np.testing.assert_array_equal(criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col), expected_criteria)

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize("img_size", [(4, 5)])
    @pytest.mark.parametrize(
        ["msk", "expected_criteria", "disp_col", "disp_row", "start_point_image", "start_point", "step"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,
                0,
                [2, 1],
                [2, 1],
                [1, 1],
                id="ROI simulation with step=[1,1]",
            ),
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,
                0,
                [1, 1],
                [0, 0],
                [2, 1],
                id="ROI simulation and margins not modulo step",
            ),
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -2,
                -1,
                [2, 1],
                [0, 0],
                [2, 1],
                id="ROI simulation and margins modulo step",
            ),
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,
                0,
                [3, 1],
                [0, 0],
                [2, 1],
                id="ROI simulation and margins greater than step",
            ),
        ],
    )
    def test_combination(self, criteria_dataarray, mask_criteria_right, expected_criteria, disp_col, disp_row):
        """
        Test apply_right_criteria_mask with step and ROI
        """
        criteria.apply_right_criteria_mask(criteria_dataarray, mask_criteria_right)

        np.testing.assert_array_equal(criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col), expected_criteria)
