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
"""Test get_criteria_dataarray function."""
import copy

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize("img_size", [(4, 5)])
class TestGetCriteriaDataarray:
    """Test get_criteria_dataarray function."""

    @pytest.fixture()
    def image_variable_disp(self, image, img_size):
        """Make image with variable disparity grids"""

        # Make so when we change image_variable_disp mask it
        # does not change image mask
        img = copy.copy(image)
        row, col = img_size

        nb_col_set = int(col / 2)
        nb_row_set = int(row / 2)

        # Get variable col disparities

        # Minimal col disparity grid is equal to:
        # [[-3, -3, -5, -5, -5]
        #  [-3, -3, -5, -5, -5]
        #  [-3, -3, -5, -5, -5]
        #  [-3, -3, -5, -5, -5]]
        img["col_disparity"].sel(band_disp="min")[:, :nb_col_set] = -3

        # Maximal col disparity grid is equal to:
        # [[ 3,  3,  1,  1,  1]
        #  [ 3,  3,  1,  1,  1]
        #  [ 3,  3,  1,  1,  1]
        #  [ 3,  3,  1,  1,  1]]
        img["col_disparity"].sel(band_disp="max")[:, nb_col_set:] = 1

        # Get variable row disparities

        # Minimal row disparity grid is equal to:
        # [[ 0,  0,  0,  0,  0]
        #  [ 0,  0,  0,  0,  0]
        #  [-1, -1, -1, -1, -1]
        #  [-1, -1, -1, -1, -1]]
        img["row_disparity"].sel(band_disp="min")[:nb_row_set, :] = 0

        # Maximal row disparity grid is equal to:
        # [[ 3,  3,  3,  3,  3]
        #  [ 3,  3,  3,  3,  3]
        #  [ 2,  2,  2,  2,  2]
        #  [ 2,  2,  2,  2,  2]]

        img["row_disparity"].sel(band_disp="max")[nb_row_set:, :] = 2

        return img

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["left_msk", "msk", "disp_col", "disp_row", "window_size", "expected_criteria"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                np.full((4, 5), 0),  # left msk
                np.full((4, 5), 0),  # right msk
                0,  # disp_col
                0,  # disp_row
                1,  # window_size
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
                id="Everything is valid",
            ),
            pytest.param(
                np.full((4, 5), 0),  # left msk
                np.full((4, 5), 0),  # right msk
                2,  # disp_col
                -1,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.P2D_DISPARITY_UNPROCESSED, Criteria.P2D_DISPARITY_UNPROCESSED, Criteria.P2D_DISPARITY_UNPROCESSED, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.VALID , Criteria.VALID , Criteria.P2D_DISPARITY_UNPROCESSED, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.VALID , Criteria.VALID , Criteria.P2D_DISPARITY_UNPROCESSED, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        # fmt: on
                    ]
                ),
                id="Criteria.P2D_DISPARITY_UNPROCESSED overcome other criteria",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 2, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_col
                1,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.P2D_LEFT_NODATA, Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.P2D_RIGHT_NODATA],
                        [Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_LEFT, Criteria.VALID],
                        [Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.P2D_RIGHT_DISPARITY_OUTSIDE , Criteria.P2D_RIGHT_DISPARITY_OUTSIDE ],
                        # fmt: on
                    ]
                ),
                id="Mix of criteria with window_size=1",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 3, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_col
                1,  # disp_row
                3,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_NODATA | Criteria.P2D_INVALID_MASK_RIGHT, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE | Criteria.P2D_INVALID_MASK_LEFT, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_BORDER , Criteria.P2D_LEFT_BORDER],
                        # fmt: on
                    ]
                ),
                id="Mix of criteria with window_size=3",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                0,  # disp_col
                1,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_LEFT_NODATA  | Criteria.P2D_INVALID_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.P2D_INVALID_MASK_LEFT, Criteria.VALID, Criteria.VALID],
                        [Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        # fmt: on
                    ]
                ),
                id="Centered invalid and no data in msk with window_size=1",
            ),
            pytest.param(
                np.full((4, 5), 0),  # left msk
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                1,  # disp_col
                1,  # disp_row
                3,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.VALID, Criteria.P2D_RIGHT_NODATA, Criteria.P2D_RIGHT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        # fmt: on
                    ]
                ),
                id="Right no data on the border and window_size=3",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 2, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -1,  # disp_col
                1,  # disp_row
                5,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        # fmt: on
                    ]
                ),
                id="Window_size=5, only Criteria.P2D_LEFT_BORDER is raised",
            ),
            pytest.param(
                np.full((4, 5), 0),  # left msk
                np.full((4, 5), 0),  # right msk
                -5,  # disp_col
                0,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE],
                        # fmt: on
                    ]
                ),
                id="Column disparity out of the image or unprocessed for all points",
            ),
            pytest.param(
                np.full((4, 5), 0),  # left msk
                np.full((4, 5), 0),  # right msk
                -5,  # disp_col
                0,  # disp_row
                3,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_DISPARITY_UNPROCESSED | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_RIGHT_DISPARITY_OUTSIDE, Criteria.P2D_LEFT_BORDER],
                        [Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_BORDER],
                        # fmt: on
                    ]
                ),
                id="Column disparity out of the image or unprocessed for all points and window_size=3",
            ),
            # pylint: enable=line-too-long
        ],
    )
    def test_get_criteria_dataarray(
        self, image_variable_disp, image, left_msk, cost_volumes, disp_col, disp_row, expected_criteria
    ):
        """
        Test get_criteria_dataarray method with
        different disparities, window sizes and masks
        """

        image_variable_disp["msk"].data = left_msk

        criteria_dataarray = criteria.get_criteria_dataarray(
            left_image=image_variable_disp, right_image=image, cv=cost_volumes
        )

        np.testing.assert_array_equal(
            criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col),
            expected_criteria,
        )
