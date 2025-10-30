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
Test mask_right_no_data function.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize("img_size", [(4, 5)])
class TestMaskRightNoData:
    """Test mask_right_no_data function."""

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["valid_pixels", "no_data_mask", "msk", "disp_row", "disp_col", "subpix", "expected_criteria"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                -1,  # disp_row
                -1,  # disp_col
                1,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (4,5) with disp_row=-1 and disp_col=-1",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                1,  # disp_row
                1,  # disp_col
                1,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (4,5) with disp_row=1 and disp_col=1",
            ),
            pytest.param(
                0,  # valid_pixels
                2,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                    ]
                ),
                2,  # disp_row
                1,  # disp_col
                1,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (4,5) with disp_row=2 and disp_col=1",
            ),
            pytest.param(
                4,  # valid_pixels
                3,  # no_data_mask
                np.array(  # msk
                    [
                        [4, 4, 4, 4, 4],
                        [4, 3, 4, 4, 4],
                        [3, 3, 4, 4, 4],
                        [3, 3, 4, 4, 4],
                    ]
                ),
                1,  # disp_row
                -2,  # disp_col
                1,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Multiple no data points with disp_row=2 and disp_col=1, no_data_mask=4, valid_pixels=3",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.5,  # disp_row
                -1.5,  # disp_col
                2,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (3,3) with disp_row=2.5 and disp_col=-1.5, subpix=2",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                0,  # disp_row
                -3.5,  # disp_col
                2,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (3,0) with disp_row=0 and disp_col=-3.5, subpix=2",
            ),
            pytest.param(
                0,  # valid_pixels
                3,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -0.75,  # disp_row
                2.25,  # disp_col
                4,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (2,3) with disp_row=-0.75 and disp_col=2.25, subpix=4",
            ),
            pytest.param(
                2,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2],
                        [2, 2, 2, 1, 2],
                        [2, 2, 2, 2, 2],
                    ]
                ),
                1.75,  # disp_row
                1,  # disp_col
                4,  # subpix
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="No data point at position (1,2) with disp_row=1.75 and disp_col=1, subpix=4",
            ),
        ],
        # pylint: enable=line-too-long
    )
    def test_window_size_1(self, image, criteria_dataarray, disp_row, disp_col, spline_order, expected_criteria):
        """Test some disparity couples with a window size of 1."""

        criteria.mask_right_no_data(image, 1, criteria_dataarray, spline_order)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["valid_pixels", "no_data_mask", "msk", "disp_row", "disp_col", "subpix", "expected_criteria"],
        # pylint: disable=line-too-long
        [
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                -1,  # disp_col
                1,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=-1 and disp_col=-1, subpix=1",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                1,  # disp_col
                1,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=-1 and disp_col=1, subpix=1",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1,  # disp_row
                1,  # disp_col
                1,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=1 and disp_col=1, subpix=1",
            ),
            pytest.param(
                0,  # valid_pixels
                3,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, 1, 3, 0],
                        [0, 0, 4, 0, 0],
                    ]
                ),
                1,  # disp_row
                1,  # disp_col
                1,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=1 and disp_col=1, subpix=1, no_data=3",
            ),
            pytest.param(
                4,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4],
                        [4, 1, 4, 4, 4],
                        [4, 4, 4, 4, 4],
                    ]
                ),
                3,  # disp_row
                2,  # disp_col
                1,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,2) with disp_row=3 and disp_col=2, subpix=1, valid_pixel=4",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.5,  # disp_row
                0.5,  # disp_col
                2,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=2.5 and disp_col=0.5, subpix=2",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1.5,  # disp_row
                -3.5,  # disp_col
                2,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (2,0) with disp_row=1.5 and disp_col=-3.5, subpix=2",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.25,  # disp_row
                0.75,  # disp_col
                4,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with disp_row=2.25 and disp_col=0.75, subpix=4",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -0.75,  # disp_row
                -2,  # disp_col
                4,  # subpix
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (0,0) with disp_row=-0.75 and disp_col=-2, subpix=4",
            ),
        ],
        # pylint: enable=line-too-long
    )
    def test_window_size_3(self, image, criteria_dataarray, disp_row, disp_col, spline_order, expected_criteria):
        """Test some disparity couples with a window size of 3."""

        criteria.mask_right_no_data(image, 3, criteria_dataarray, spline_order)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        [
            "valid_pixels",
            "no_data_mask",
            "msk",
            "disp_row",
            "disp_col",
            "subpix",
            "window_size",
            "spline_order",
            "expected_criteria",
        ],
        # pylint: disable=line-too-long
        [
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                -1.5,  # disp_col
                2,  # subpix
                1,  # window_size
                2,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=2, window_size=1, spline_order=2",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                -1.5,  # disp_col
                2,  # subpix
                1,  # window_size
                3,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=2, window_size=1, spline_order=3",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                -1.5,  # disp_col
                2,  # subpix
                1,  # window_size
                4,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=2, window_size=1, spline_order=4",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_row
                -1.5,  # disp_col
                2,  # subpix
                1,  # window_size
                5,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=2, window_size=1, spline_order=5",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -0.25,  # disp_row
                0.75,  # disp_col
                4,  # subpix
                3,  # window_size
                2,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=4, window_size=3, spline_order=2",
            ),
            pytest.param(
                0,  # valid_pixels
                1,  # no_data_mask
                np.array(  # msk
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -0.25,  # disp_row
                0.75,  # disp_col
                4,  # subpix
                3,  # window_size
                3,  # spline_order
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="No data point at position (3,4) with subpix=4, window_size=3, spline_order=2",
            ),
        ],
        # pylint: enable=line-too-long
    )
    def test_spline_order(
        self, image, window_size, criteria_dataarray, disp_row, disp_col, spline_order, expected_criteria
    ):
        """Test some disparity couples with a different window size and spline_order."""

        criteria.mask_right_no_data(image, window_size, criteria_dataarray, spline_order)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    def test_combination(self, image, criteria_dataarray, spline_order):
        """Test that we combine with existing criteria and do not override them."""
        image["msk"].data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        criteria_dataarray.data[2, 3, ...] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

        criteria.mask_right_no_data(image, 1, criteria_dataarray, spline_order)

        assert (
            criteria_dataarray.sel(row=2, col=3, disp_row=1, disp_col=1).data
            == Criteria.P2D_RIGHT_DISPARITY_OUTSIDE | Criteria.P2D_RIGHT_NODATA
        )
