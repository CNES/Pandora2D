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

"""
Test mask_disparity_outside_right_image method.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def ground_truth_null_disparity(offset, img_size):
    """Make ground_truth of criteria dataarray for null disparity"""
    data = np.full(img_size, Criteria.VALID)
    if offset > 0:
        data[:offset, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
        data[-offset:, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
        data[:, :offset] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
        data[:, -offset:] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    return data


@pytest.fixture()
def ground_truth_first_disparity(offset, img_size):
    """
    Make ground_truth of criteria dataarray for first disparity (disp_col=-5 and disp_row=-1)

    Example for window_size = 3 -> offset = 1, disp_col=-5 & disp_row=-1 & img_size = (10, 13)
    data = ([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0]
        ])

    Example for window_size = 5 -> offset = 2, disp_col=-5 & disp_row=-1 & img_size = (10, 13)
    data = ([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        ])
    """
    data = np.full(img_size, Criteria.VALID)
    # Update row
    first_row_disparity = -1
    delta_row_start = offset + abs(first_row_disparity)
    delta_row_end = offset + first_row_disparity
    data[:delta_row_start, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    if delta_row_end > 0:
        data[-delta_row_end:, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    # Udpate col
    first_col_disparity = -5
    delta_col_start = offset + abs(first_col_disparity)
    delta_col_end = offset + first_col_disparity
    data[:, :delta_col_start] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    if delta_col_end > 0:
        data[:, -delta_col_end:] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    return data


@pytest.mark.parametrize(
    "offset",
    [
        pytest.param(0),
        pytest.param(1),
        pytest.param(2),
        pytest.param(3),
        pytest.param(49, id="offset > dimension"),
    ],
)
def test_nominal(offset, image, criteria_dataarray, ground_truth_null_disparity, ground_truth_first_disparity):
    """
    Test mask_disparity_outside_right_image
    """
    criteria.mask_disparity_outside_right_image(image, offset, criteria_dataarray)

    np.testing.assert_array_equal(criteria_dataarray.values[:, :, 1, 5], ground_truth_null_disparity)
    np.testing.assert_array_equal(criteria_dataarray.values[:, :, 0, 0], ground_truth_first_disparity)
