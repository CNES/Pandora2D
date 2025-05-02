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
        ["no_data_mask", "msk", "disp_row", "disp_col", "subpix", "expected_criteria"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                -1,
                -1,
                1,
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
                id="Disp -1 -1 - Pos (3,4)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                -1,
                1,
                1,
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
                id="Disp -1 1 - Pos (3,4)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                1,
                1,
                1,
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
                id="Disp 1 1 - Pos (3,4)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                2,
                1,
                1,
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
                id="Disp 2 1 - Pos (3,4)",
            ),
            pytest.param(
                2,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                    ]
                ),
                2,
                1,
                1,
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
                id="Disp 2 1 - other no_data_mask",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.5,
                -1.5,
                2,
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Disp 2.5 -1.5 - Pos (2,2), subpix=2",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                0,
                -3.5,
                2,
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Disp 0 -3.5 - Pos (2,0), subpix=2",
            ),
            pytest.param(
                3,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                0.75,
                -2.25,
                4,
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Disp 0.75 -2.25 - Pos (1,2), no_data_mask=3, subpix=4",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1.75,
                1,
                4,
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Disp 1.75 1 - Pos (1,2), subpix=4",
            ),
        ],
        # pylint: enable=line-too-long
    )
    def test_window_size_1(self, image, criteria_dataarray, disp_row, disp_col, expected_criteria):
        """Test some disparity couples with a window size of 1."""

        criteria.mask_right_no_data(image, 1, criteria_dataarray)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["no_data_mask", "msk", "disp_row", "disp_col", "subpix", "expected_criteria"],
        # pylint: disable=line-too-long
        [
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,
                -1,
                1,
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
                id="Disp -1 -1 - Pos (2,3)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,
                1,
                1,
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
                id="Disp -1 1 - Pos (2,3)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1,
                1,
                1,
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
                id="Disp 1 1 - Pos (2,3)",
            ),
            pytest.param(
                3,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, 1, 3, 0],
                        [0, 0, 4, 0, 0],
                    ]
                ),
                1,
                1,
                1,
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
                id="Disp 1 1 - Pos (2,3) - other no_data_mask",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2,
                1,
                1,
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="Disp 2 1 - Pos (2,3)",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.5,
                0.5,
                2,
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="Disp 2.5 0.5 - Pos (2,3), subpix=2",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1.5,
                -3.5,
                2,
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
                id="Disp 1.5 -3.5 - Pos (2,0), subpix=2",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2.25,
                0.75,
                4,
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="Disp 2.25 0.75 - Pos (2,3), subpix=4",
            ),
            pytest.param(
                1,
                np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -0.75,
                -2.25,
                4,
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
                id="Disp -0.75 -2.25 - Pos (0,0), subpix=4",
            ),
        ],
        # pylint: enable=line-too-long
    )
    def test_window_size_3(self, image, criteria_dataarray, disp_row, disp_col, expected_criteria):
        """Test some disparity couples with a window size of 3."""

        criteria.mask_right_no_data(image, 3, criteria_dataarray)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    def test_combination(self, image, criteria_dataarray):
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

        criteria.mask_right_no_data(image, 1, criteria_dataarray)

        assert (
            criteria_dataarray.sel(row=2, col=3, disp_row=1, disp_col=1).data
            == Criteria.P2D_RIGHT_DISPARITY_OUTSIDE | Criteria.P2D_RIGHT_NODATA
        )
