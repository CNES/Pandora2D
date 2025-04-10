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
Test mask_right_invalid function.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize("img_size", [(4, 5)])
class TestMaskRightInvalid:
    """Test mask_right_invalid function."""

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["valid_pixels", "no_data_mask", "msk", "expected_criteria", "disp_col", "disp_row", "subpix"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT],
                        # fmt: on
                    ]
                ),
                -1,  # disp_col
                -1,  # disp_row
                1,  # subpix
                id="Invalid point at center of right mask with disp_row=-1 and disp_col=-1",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                2,  # disp_col
                1,  # disp_row
                1,  # subpix
                id="Invalid point at center of right mask with disp_row=2 and disp_col=1",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -1,  # disp_col
                -1,  # disp_row
                1,  # subpix
                id="Invalid point at right bottom corner of right mask with disp_row=-1 and disp_col=-1",
            ),
            pytest.param(
                0,
                1,
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
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -1,  # disp_col
                -1,  # disp_row1, # subpix
                1,  # subpix
                id="Invalid point at center of right mask with disp_row=-1 and disp_col=-1",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 4, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,  # disp_col
                -1,  # disp_row
                1,  # subpix
                id="Invalid point at center of right mask with disp_row=-1 and disp_col=0",
            ),
            pytest.param(
                3,
                4,
                np.array(  # msk
                    [
                        [3, 3, 3, 3, 3],
                        [3, 3, 0, 3, 4],
                        [3, 3, 4, 3, 3],
                        [3, 4, 3, 3, 3],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,  # disp_col
                -1,  # disp_row
                1,  # subpix
                id="Invalid point at center of right mask with no_data_mask=4, valid_pixels=3, disp_row=-1 and disp_col=0",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                1.5,  # disp_col
                -0.5,  # disp_row
                2,  # subpix
                id="Invalid point at center of right mask with subpix=2, disp_row=-0.5 and disp_col=1.5",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
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
                0,  # disp_col
                0,  # disp_row
                2,  # subpix
                id="Invalid point at center of right mask with subpix=2, disp_row=0 and disp_col=0",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
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
                0.5,  # disp_col
                0.5,  # disp_row
                2,  # subpix
                id="Invalid point at center of right mask with subpix=2, disp_row=0.5 and disp_col=0.5",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                3,  # disp_col
                2.5,  # disp_row
                2,  # subpix
                id="Invalid point at center of right mask with subpix=2, disp_row=2.5 and disp_col=3",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 3],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -1.5,  # disp_col
                -1,  # disp_row
                2,  # subpix
                id="Invalid point at right bottom corner of right mask with subpix=2, disp_row=-1 and disp_col=-1.5",
            ),
            pytest.param(
                3,
                4,
                np.array(  # msk
                    [
                        [3, 3, 3, 3, 3],
                        [3, 3, 0, 3, 4],
                        [3, 3, 4, 3, 3],
                        [3, 4, 3, 3, 3],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0.5,  # disp_col
                -1,  # disp_row
                2,  # subpix
                id="Invalid point at center of right mask with subpix=2, no_data_mask=4, valid_pixels=3, disp_row=-1 and disp_col=0.5",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                1.25,  # disp_col
                -0.75,  # disp_row
                4,  # subpix
                id="Invalid point at center of right mask with supix=4, disp_row=-0.75 and disp_col=1.25",
            ),
            pytest.param(
                0,
                1,
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                2.75,  # disp_col
                1.5,  # disp_row
                4,  # subpix
                id="Invalid point at center of right mask with supix=4, disp_row=2.75 and disp_col=1.5",
            ),
            pytest.param(
                3,
                4,
                np.array(  # msk
                    [
                        [3, 3, 3, 3, 3],
                        [3, 3, 0, 3, 4],
                        [3, 3, 4, 3, 3],
                        [3, 4, 3, 3, 3],
                    ]
                ),
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                1,  # disp_col
                -0.25,  # disp_row
                4,  # subpix
                id="Invalid point at center of right mask with subpix=4, no_data_mask=4, valid_pixels=3, disp_row=-0.25 and disp_col=1",
            ),
            # pylint: enable=line-too-long
        ],
    )
    def test_mask_invalid_right(self, image, criteria_dataarray, expected_criteria, disp_col, disp_row):
        """
        Test that mask_invalid_right method raises criteria P2D_INVALID_MASK_RIGHT
        for points whose value is neither valid_pixels or no_data_mask when we shift it by its disparity.
        """

        criteria.mask_right_invalid(image, criteria_dataarray)

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["msk", "disp_col", "disp_row"],
        [
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -2,  # disp_col
                -1,  # disp_row
                id="Invalid point at center of right mask with disp_row=-1 and disp_col=-2",
            ),
            pytest.param(
                np.array(  # msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                    ]
                ),
                1,  # disp_col
                1,  # disp_row
                id="Invalid point at right bottom corner of right mask with disp_row=1 and disp_col=1",
            ),
        ],
    )
    def test_combination(self, image, criteria_dataarray, disp_col, disp_row):
        """
        Test that we combine Criteria.P2D_INVALID_MASK_RIGHT
        with existing criteria and do not override them.
        """

        criteria_dataarray.data[2, 3, ...] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

        criteria.mask_right_invalid(image, criteria_dataarray)

        assert (
            criteria_dataarray.sel(row=2, col=3, disp_row=disp_row, disp_col=disp_col).data
            == Criteria.P2D_RIGHT_DISPARITY_OUTSIDE | Criteria.P2D_INVALID_MASK_RIGHT
        )
