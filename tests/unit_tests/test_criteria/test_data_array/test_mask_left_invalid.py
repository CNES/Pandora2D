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
Test mask_left_invalid function.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize(
    ["invalid_position"],
    [
        pytest.param((2, 2)),
        pytest.param((0, 0)),
        pytest.param((0, 2)),
        pytest.param((9, 12)),
        pytest.param((4, 5)),
    ],
)
def test_mask_left_invalid(img_size, image, criteria_dataarray, invalid_position):
    """
    Test that mask_invalid_left method raises criteria P2D_INVALID_MASK_LEFT
    for points whose value is neither valid_pixels or no_data_mask.
    """
    invalid_row_position, invalid_col_position = invalid_position

    # We put 2 in img_left msk because it is different from valid_pixels=0 and no_data_mask=1
    image["msk"][invalid_row_position, invalid_col_position] = 2

    expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
    expected_criteria_data[invalid_row_position, invalid_col_position, ...] = Criteria.P2D_INVALID_MASK_LEFT

    criteria.mask_left_invalid(image, criteria_dataarray)

    np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)


@pytest.mark.parametrize(
    ["invalid_position"],
    [
        pytest.param((2, 2)),
        pytest.param((0, 0)),
        pytest.param((0, 2)),
        pytest.param((9, 12)),
        pytest.param((4, 5)),
    ],
)
def test_add_to_existing(img_size, image, criteria_dataarray, invalid_position):
    """Test we do not override existing criteria but combine it."""
    invalid_row_position, invalid_col_position = invalid_position

    # We put 2 in img_left msk because it is different from valid_pixels=0 and no_data_mask=1
    image["msk"][invalid_row_position, invalid_col_position] = 2

    criteria_dataarray.data[invalid_row_position, invalid_col_position, ...] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

    expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
    expected_criteria_data[invalid_row_position, invalid_col_position, ...] = (
        Criteria.P2D_INVALID_MASK_LEFT | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    )

    criteria.mask_left_invalid(image, criteria_dataarray)

    np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)
