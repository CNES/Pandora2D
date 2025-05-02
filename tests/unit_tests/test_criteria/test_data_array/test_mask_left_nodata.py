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
Test mask_left_no_data function.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize("img_size", [(5, 6)])
class TestMaskLeftNoData:
    """Test mask_left_no_data function."""

    @pytest.mark.parametrize(
        ["no_data_position", "window_size", "row_slice", "col_slice"],
        [
            pytest.param((2, 2), 1, 2, 2),
            pytest.param((2, 2), 3, np.s_[1:4], np.s_[1:4]),
            pytest.param((0, 2), 1, 0, 2),
            pytest.param((0, 2), 3, np.s_[:2], np.s_[1:4]),
            pytest.param((4, 5), 3, np.s_[-2:], np.s_[-2:]),
        ],
    )
    def test_add_criteria_to_all_valid(
        self, img_size, image, criteria_dataarray, no_data_position, window_size, row_slice, col_slice
    ):
        """Test add to a previously VALID criteria."""
        no_data_row_position, no_data_col_position = no_data_position

        image["msk"][no_data_row_position, no_data_col_position] = image.attrs["no_data_mask"]

        expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
        expected_criteria_data[row_slice, col_slice, ...] = Criteria.P2D_LEFT_NODATA

        criteria.mask_left_no_data(image, window_size, criteria_dataarray)

        np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)

    @pytest.mark.parametrize(
        ["no_data_position", "window_size", "row_slice", "col_slice"],
        [
            pytest.param((2, 2), 1, 2, 2),
            pytest.param((2, 2), 3, np.s_[1:4], np.s_[1:4]),
            pytest.param((0, 2), 1, 0, 2),
            pytest.param((0, 2), 3, np.s_[:2], np.s_[1:4]),
            pytest.param((4, 5), 3, np.s_[-2:], np.s_[-2:]),
        ],
    )
    def test_add_to_existing(
        self, img_size, image, criteria_dataarray, no_data_position, window_size, row_slice, col_slice
    ):
        """Test we do not override existing criteria but combine it."""
        no_data_row_position, no_data_col_position = no_data_position

        image["msk"][no_data_row_position, no_data_col_position] = image.attrs["no_data_mask"]

        criteria_dataarray.data[no_data_row_position, no_data_col_position, ...] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

        expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
        expected_criteria_data[row_slice, col_slice, ...] = Criteria.P2D_LEFT_NODATA
        expected_criteria_data[no_data_row_position, no_data_col_position, ...] = (
            Criteria.P2D_LEFT_NODATA | Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
        )

        criteria.mask_left_no_data(image, window_size, criteria_dataarray)

        np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)
