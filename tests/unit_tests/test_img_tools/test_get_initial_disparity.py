# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2026 CS GROUP France
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
Test get_initial_disparity method.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora2d import img_tools


class TestGetInitialDisparity:
    """Test get_initial_disparity behavior."""

    @pytest.fixture
    def second_correct_grid_data(self, second_correct_grid_shape):
        """second_correct_grid_data override to include inf and nan."""
        data = np.full(second_correct_grid_shape, 5.0)
        data[:, 1::4] = -21
        data[:, 2::4] = -np.inf
        data[:, 3::4] = np.nan
        return data

    @pytest.fixture
    def centered_roi(self, left_img_shape):
        """Create a ROI that excludes a 2-pixel border on each side."""
        height, width = left_img_shape
        return {
            "col": {"first": 2, "last": width - 3},
            "row": {"first": 2, "last": height - 3},
            "margins": [2, 2, 2, 2],
        }

    @pytest.mark.parametrize(
        ["second_correct_grid_shape", "nodata", "expected"], [((1, 5), None, [[[5, -21, np.nan, np.nan, 5]]])]
    )
    def test_get_initial_disparity_str(
        self, create_disparity_grid_fixture, second_correct_grid_data, second_correct_grid_shape, nodata, expected
    ):
        """Test invalid values are replaced by NaNs."""
        disparity = create_disparity_grid_fixture(second_correct_grid_data, 2, "disparity.tiff", nodata=nodata)

        result = img_tools.get_initial_disparity(disparity)

        np.testing.assert_array_equal(result, expected)

    def test_get_initial_disparity_int(self):
        """Test init is returned."""
        disparity = {"init": 1, "range": 2}

        result = img_tools.get_initial_disparity(disparity)

        assert result == 1

    def test_get_initial_disparity_roi_is_none(self, border_outliers_grid, border_outliers_grid_data):
        """When roi=None, full disparity grid is read including border outliers."""
        result = img_tools.get_initial_disparity(border_outliers_grid, roi=None)

        assert result.shape == (1, *border_outliers_grid_data.shape)

        expected = border_outliers_grid_data[np.newaxis, ...]
        np.testing.assert_array_equal(result, expected)

    def test_get_initial_disparity_roi_is_not_none(self, border_outliers_grid, left_img_shape, centered_roi):
        """When roi is not None, only ROI window pixels are read."""
        result = img_tools.get_initial_disparity(border_outliers_grid, roi=centered_roi)

        roi_height = centered_roi["row"]["last"] - centered_roi["row"]["first"] + 1
        roi_width = centered_roi["col"]["last"] - centered_roi["col"]["first"] + 1
        expected_shape = (1, roi_height, roi_width)
        assert result.shape == expected_shape
        # The ROI window excludes border outliers, so only interior 1.0 values are read.
        np.testing.assert_array_equal(result, np.ones(expected_shape, dtype=np.float32))

    def test_get_initial_disparity_from_previous_run_reads_full_raster(
        self, border_outliers_grid, left_img_shape, centered_roi
    ):
        """
        When from_previous_run=True, the full raster is read even if a ROI is provided.

        This covers the re-entrance case where the raster is already cropped to the ROI zone
        and stored in local coordinates: applying the global ROI as a window offset would be wrong.
        """
        height, width = left_img_shape

        result = img_tools.get_initial_disparity(border_outliers_grid, roi=centered_roi, from_previous_run=True)

        assert result.shape == (1, height, width)
        assert np.any(result == 100.0)
