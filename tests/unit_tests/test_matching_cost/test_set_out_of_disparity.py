# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
Test set_out_of_disparity methods from Matching cost
"""

import pytest
import numpy as np
import xarray as xr

from pandora2d.matching_cost.base import (
    set_out_of_col_disparity_range_to_other_value,
    set_out_of_row_disparity_range_to_other_value,
)


class TestSetOutOfDisparity:
    """Test effect of disparity grids."""

    @pytest.fixture()
    def disp_coords(self):
        return "disp_row"

    @pytest.fixture()
    def init_value(self):
        return 0.0

    @pytest.fixture()
    def range_col(self):
        return np.arange(4)

    @pytest.fixture()
    def range_row(self):
        return np.arange(5)

    @pytest.fixture()
    def disp_range_col(self):
        return np.arange(2, 2 + 7)

    @pytest.fixture()
    def disp_range_row(self):
        return np.arange(-5, -5 + 6)

    @pytest.fixture()
    def dataset(self, range_row, range_col, disp_range_col, disp_range_row, init_value, disp_coords):
        """make a xarray dataset and disparity grids"""
        xarray = xr.DataArray(
            np.full((5, 4, 6, 7), init_value),
            coords={
                "row": range_row,
                "col": range_col,
                "disp_row": disp_range_row,
                "disp_col": disp_range_col,
            },
            dims=["row", "col", "disp_row", "disp_col"],
        )

        xarray.attrs = {"col_disparity_source": [2, 8], "row_disparity_source": [-5, 0]}
        min_disp_grid = np.full((xarray.sizes["row"], xarray.sizes["col"]), xarray.coords[disp_coords].data[0])
        max_disp_grid = np.full((xarray.sizes["row"], xarray.sizes["col"]), xarray.coords[disp_coords].data[-1])
        return xarray, min_disp_grid, max_disp_grid

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, np.nan],
            [0.0, 1],
            [0.0, -1],
            [0.0, np.inf],
        ],
    )
    def test_homogeneous_row_grids(self, dataset, value):
        """With grids set to extreme disparities, cost_volumes should be left untouched."""
        # As set_out_of_row_disparity_range_to_other_value modify cost_volumes in place we do a copy to be able
        # to make the comparison later.
        array, min_disp_grid, max_disp_grid = dataset
        make_array_copy = array.copy(deep=True)
        set_out_of_row_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["row_disparity_source"]
        )

        xr.testing.assert_equal(array, make_array_copy)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, np.nan],
            [0.0, 10],
            [0.0, -10],
            [0.0, np.inf],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_homogeneous_col_grids(self, dataset, value):
        """With grids set to extreme disparities, cost_volumes should be left untouched."""
        # As set_out_of_col_disparity_range_to_other_value modify cost_volumes in place we do a copy to be able
        # to make the comparison later.
        array, min_disp_grid, max_disp_grid = dataset
        make_array_copy = array.copy(deep=True)
        set_out_of_col_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["col_disparity_source"]
        )

        xr.testing.assert_equal(array, make_array_copy)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    def test_variable_min_row(self, dataset, value, disp_coords, init_value):
        """Check special value below min disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[::2] = array.coords[disp_coords].data[min_disp_index]

        set_out_of_row_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["row_disparity_source"]
        )

        expected_value = array.data[::2, :, :min_disp_index, :]
        expected_init_value_on_odd_lines = array.data[1::2, ...]
        expected_init_value_on_even_lines = array.data[::2, :, min_disp_index:, :]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_lines == init_value)
        assert np.all(expected_init_value_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_min_col(self, dataset, value, disp_coords, init_value):
        """Check special value below min disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[:, ::2] = array.coords[disp_coords].data[min_disp_index]

        set_out_of_col_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["col_disparity_source"]
        )

        expected_value = array.data[:, ::2, :, :min_disp_index]
        expected_init_value_on_odd_columns = array.data[:, 1::2, ...]
        expected_init_value_on_even_columns = array.data[:, ::2, :, min_disp_index:]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_columns == init_value)
        assert np.all(expected_init_value_on_even_columns == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    def test_variable_max_row(self, dataset, value, disp_coords, init_value):
        """Check special value above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        max_disp_index = 1
        max_disp_grid[::2] = array.coords[disp_coords].data[max_disp_index]

        set_out_of_row_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["row_disparity_source"]
        )

        expected_value = array.data[::2, :, (max_disp_index + 1) :, :]
        expected_init_value_on_odd_lines = array.data[1::2, ...]
        expected_init_value_on_even_lines = array.data[::2, :, : (max_disp_index + 1), :]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_lines == init_value)
        assert np.all(expected_init_value_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_max_col(self, dataset, value, disp_coords, init_value):
        """Check special value above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        max_disp_index = 1
        max_disp_grid[:, ::2] = array.coords[disp_coords].data[max_disp_index]

        set_out_of_col_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["col_disparity_source"]
        )

        expected_value = array.data[:, ::2, :, (max_disp_index + 1) :]
        expected_init_value_on_odd_columns = array.data[:, 1::2, ...]
        expected_init_value_on_even_columns = array.data[:, ::2, :, : (max_disp_index + 1)]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_columns == init_value)
        assert np.all(expected_init_value_on_even_columns == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    def test_variable_min_and_max_row(self, dataset, value, disp_coords, init_value):
        """Check special value below min and above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[::2] = array.coords[disp_coords].data[min_disp_index]
        max_disp_index = 2
        max_disp_grid[::2] = array.coords[disp_coords].data[max_disp_index]

        set_out_of_row_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["row_disparity_source"]
        )

        expected_below_min = array.data[::2, :, :min_disp_index, :]
        expected_above_max = array.data[::2, :, (max_disp_index + 1) :, :]
        expected_init_value_on_odd_lines = array.data[1::2, ...]
        expected_init_value_on_even_lines = array.data[::2, :, min_disp_index : (max_disp_index + 1), :]

        assert np.all(expected_below_min == value)
        assert np.all(expected_above_max == value)
        assert np.all(expected_init_value_on_odd_lines == init_value)
        assert np.all(expected_init_value_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_min_and_max_col(self, dataset, value, disp_coords, init_value):
        """Check special value below min and above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[:, ::2] = array.coords[disp_coords].data[min_disp_index]
        max_disp_index = 2
        max_disp_grid[:, ::2] = array.coords[disp_coords].data[max_disp_index]

        set_out_of_col_disparity_range_to_other_value(
            array, min_disp_grid, max_disp_grid, value, array.attrs["col_disparity_source"]
        )

        expected_below_min = array.data[:, ::2, :, :min_disp_index]
        expected_above_max = array.data[:, ::2, :, (max_disp_index + 1) :]
        expected_init_value_on_odd_columns = array.data[:, 1::2, ...]
        expected_init_value_on_even_columns = array.data[:, ::2, :, min_disp_index : (max_disp_index + 1)]

        assert np.all(expected_below_min == value)
        assert np.all(expected_above_max == value)
        assert np.all(expected_init_value_on_odd_columns == init_value)
        assert np.all(expected_init_value_on_even_columns == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    def test_variable_min_row_with_disp_margins(self, dataset, value, disp_coords, init_value):
        """
        Check special value below min local disparities but above min disparity margins.

        Here, we test the method's behavior when the minimum row disparity grid is -3 on even rows and -4 on odd rows,
        but the first row disparity of the dataset is -5 (equivalent to a disparity margin).

        In this case, the output of the method should have the even lines at value for disp_row=-4
        but still at init_value for disp_row=-5.
        """
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_grid = np.full((array.sizes["row"], array.sizes["col"]), array.coords[disp_coords].data[1])
        min_disp_index = 2
        min_disp_grid[::2] = array.coords[disp_coords].data[min_disp_index]

        set_out_of_row_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value, [-4, 0])

        expected_value = array.data[::2, :, min_disp_index - 1 : min_disp_index, :]
        expected_init_value_on_odd_lines = array.data[1::2, ...]
        expected_init_value_on_even_lines = array.data[::2, :, min_disp_index:, :]
        expected_init_value_on_disp_margins = array.data[:, :, 0, :]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_lines == init_value)
        assert np.all(expected_init_value_on_even_lines == init_value)
        assert np.all(expected_init_value_on_disp_margins == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_max_col_with_disp_margins(self, dataset, value, disp_coords, init_value):
        """
        Check special value above max local disparities but below max disparity margins.

        Here, we test the method's behavior when the maximum col disparity grid is 4 on even columns and 6 on odd rows,
        but the tow last row disparities of the dataset are 7 and 8 (equivalent to disparity margins).

        In this case, the output of the method should have the even lines at value for disp_col=5 and disp_col=6
        but still at init_value for disp_col=7 and disp_col=8.
        """
        array, min_disp_grid, max_disp_grid = dataset
        max_disp_grid = np.full((array.sizes["row"], array.sizes["col"]), array.coords[disp_coords].data[-3])
        max_disp_index = 2
        max_disp_grid[:, ::2] = array.coords[disp_coords].data[max_disp_index]

        set_out_of_col_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value, [2, 6])

        expected_value = array.data[:, ::2, :, (max_disp_index + 1) : (max_disp_index + 3)]
        expected_init_value_on_odd_columns = array.data[:, 1::2, ...]
        expected_init_value_on_even_columns = array.data[:, ::2, :, : (max_disp_index + 1)]
        expected_init_value_on_disp_margins = array.data[:, :, :, 5:]

        assert np.all(expected_value == value)
        assert np.all(expected_init_value_on_odd_columns == init_value)
        assert np.all(expected_init_value_on_even_columns == init_value)
        assert np.all(expected_init_value_on_disp_margins == init_value)
