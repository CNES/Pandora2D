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
Test methods from criteria.py file
"""
# pylint: disable=redefined-outer-name

import pytest
import numpy as np

from pandora2d import criteria
from pandora2d.constants import Criteria


class TestCriteria:
    """Test the Criteria Enum."""

    def test_can_be_stored_in_uint8_np_array(self):
        """Criteria can be stored in uint8 numpy array."""
        result = np.array([Criteria.VALID, Criteria.P2D_LEFT_BORDER], dtype=np.uint8)
        assert result.dtype == np.uint8

    def test_is_in(self):
        """Test method to see if a Criteria is part of an array."""
        data = np.array(
            [
                Criteria.VALID,
                Criteria.P2D_LEFT_BORDER,
                Criteria.P2D_LEFT_BORDER | Criteria.P2D_PEAK_ON_EDGE,
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(Criteria.P2D_LEFT_BORDER.is_in(data), [False, True, True])
        np.testing.assert_array_equal(Criteria.P2D_PEAK_ON_EDGE.is_in(data), [False, False, True])


class TestFlagArray:
    """Test flag array."""

    @pytest.fixture(scope="class")
    def flag_array(self):
        return criteria.FlagArray(
            [
                Criteria.P2D_PEAK_ON_EDGE,
                Criteria.P2D_RIGHT_NODATA,
            ],
            Criteria,
        )

    def test_default_dtype(self, flag_array):
        assert flag_array.dtype == np.uint8

    def test_repr(self, flag_array):
        """Test repr."""
        prefix = "FlagArray<Criteria>"
        prefix_offset = " " * (len(prefix) + 1)
        expected = (
            f"{prefix}([<P2D_PEAK_ON_EDGE: "
            f"{Criteria.P2D_PEAK_ON_EDGE.value}>,\n{prefix_offset}"
            f"<P2D_RIGHT_NODATA: "
            f"{Criteria.P2D_RIGHT_NODATA.value}>], "
            f"dtype=uint8)"
        )
        assert repr(flag_array) == expected


class TestAllocateCriteriaDataset:
    """Test create a criteria xarray.Dataset."""

    @pytest.mark.parametrize(
        ["value", "data_type"],
        [
            [0, None],
            [0, np.uint8],
            [np.nan, np.float32],
            [Criteria.VALID, None],
            [Criteria.VALID.value, np.uint16],
        ],
    )
    def test_nominal_case(self, cost_volumes, value, data_type):
        """Test allocate a criteria dataarray with correct cost_volumes, value and data_type."""
        criteria_dataarray = criteria.allocate_criteria_dataarray(cost_volumes, value, data_type)

        assert criteria_dataarray.shape == cost_volumes.cost_volumes.shape

    @pytest.mark.parametrize("value", [0, Criteria.VALID])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test_with_subpix(self, cost_volumes, value, subpix, img_size, disparity_cfg):
        """Test allocate a criteria dataarray with correct cost_volumes, value and data_type."""
        criteria_dataarray = criteria.allocate_criteria_dataarray(cost_volumes, value, None)

        row, col = img_size
        row_disparity, col_disparity = disparity_cfg
        nb_col_disp = 2 * col_disparity["range"] * subpix + 1
        nb_row_disp = 2 * row_disparity["range"] * subpix + 1

        assert criteria_dataarray.shape == cost_volumes.cost_volumes.shape
        assert criteria_dataarray.shape == (row, col, nb_row_disp, nb_col_disp)
