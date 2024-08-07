# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Test criteria dataset method 
"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np
import xarray as xr

from pandora2d import matching_cost, criteria
from pandora2d.constants import Criteria
from pandora2d.img_tools import add_disparity_grid


@pytest.fixture()
def img_size():
    row = 10
    col = 13
    return (row, col)


@pytest.fixture()
def disparity_cfg():
    return {"init": 1, "range": 2}, {"init": -1, "range": 4}


@pytest.fixture()
def img_left(img_size, disparity_cfg):
    """Make left image"""
    row, col = img_size
    row_disparity, col_disparity = disparity_cfg
    data = np.random.uniform(0, row * col, (row, col))

    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        attrs={
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "invalid_disparity": np.nan,
        },
    ).pipe(add_disparity_grid, col_disparity, row_disparity)


@pytest.fixture()
def window_size():
    return 1


@pytest.fixture()
def subpix():
    return 1


@pytest.fixture()
def matching_cost_cfg(window_size, subpix):
    return {"matching_cost_method": "ssd", "window_size": window_size, "subpix": subpix}


@pytest.fixture()
def cost_volumes(matching_cost_cfg, img_left):
    """Compute a cost_volumes"""
    matching_cost_ = matching_cost.MatchingCost(matching_cost_cfg)

    matching_cost_.allocate_cost_volume_pandora(img_left=img_left, img_right=img_left, cfg=matching_cost_cfg)
    return matching_cost_.compute_cost_volumes(img_left=img_left, img_right=img_left)


@pytest.fixture()
def criteria_dataset(cost_volumes):
    return criteria.allocate_criteria_dataset(cost_volumes, Criteria.VALID, None)


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
        """Test allocate a criteria dataset with correct cost_volumes, value and data_type."""
        criteria_dataset = criteria.allocate_criteria_dataset(cost_volumes, value, data_type)

        assert criteria_dataset.criteria.shape == cost_volumes.cost_volumes.shape

    @pytest.mark.parametrize("value", [0, Criteria.VALID])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test_with_subpix(self, cost_volumes, value, subpix, img_size, disparity_cfg):
        """Test allocate a criteria dataset with correct cost_volumes, value and data_type."""
        criteria_dataset = criteria.allocate_criteria_dataset(cost_volumes, value, None)

        row, col = img_size
        row_disparity, col_disparity = disparity_cfg
        nb_col_disp = 2 * col_disparity["range"] * subpix + 1
        nb_row_disp = 2 * row_disparity["range"] * subpix + 1

        assert criteria_dataset.criteria.shape == cost_volumes.cost_volumes.shape
        assert criteria_dataset.criteria.shape == (row, col, nb_col_disp, nb_row_disp)


class TestSetUnprocessedDisparity:
    """Test create a criteria xarray.Dataset."""

    @pytest.fixture()
    def grid_min_col(self, img_left):
        return img_left["col_disparity"].sel(band_disp="min")

    @pytest.fixture()
    def grid_max_col(self, img_left):
        return img_left["col_disparity"].sel(band_disp="max")

    @pytest.fixture()
    def grid_min_row(self, img_left):
        return img_left["row_disparity"].sel(band_disp="min")

    @pytest.fixture()
    def grid_max_row(self, img_left):
        return img_left["row_disparity"].sel(band_disp="max")

    def test_homogeneous_grids(self, criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row):
        """With uniform grids"""
        make_criteria_copy = criteria_dataset.copy(deep=True)
        criteria.set_unprocessed_disp(criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        xr.testing.assert_equal(criteria_dataset, make_criteria_copy)

    def test_variable_col_disparity(
        self, criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size
    ):
        """With variable column disparity grids"""
        _, col = img_size
        nb_col_set = int(col / 2)
        grid_min_col[:, :nb_col_set] = criteria_dataset.coords["disp_col"].data[1]
        grid_max_col[:, nb_col_set:] = criteria_dataset.coords["disp_col"].data[-2]

        criteria.set_unprocessed_disp(criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        assert np.all(
            criteria_dataset.criteria.data[:, :nb_col_set, 0, :] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataset.criteria.data[:, nb_col_set:, 0, :] == Criteria.VALID)
        assert np.all(
            criteria_dataset.criteria.data[:, nb_col_set:, -1, :] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataset.criteria.data[:, :nb_col_set, -1, :] == Criteria.VALID)

    def test_variable_row_disparity(
        self, criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size
    ):
        """With variable row disparity grids"""
        row, _ = img_size
        nb_row_set = int(row / 2)
        grid_min_row[:nb_row_set, :] = criteria_dataset.coords["disp_row"].data[1]
        grid_max_row[nb_row_set:, :] = criteria_dataset.coords["disp_row"].data[-2]

        criteria.set_unprocessed_disp(criteria_dataset, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        assert np.all(
            criteria_dataset.criteria.data[:nb_row_set, :, :, 0] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataset.criteria.data[nb_row_set:, :, :, 0] == Criteria.VALID)
        assert np.all(
            criteria_dataset.criteria.data[nb_row_set:, :, :, -1] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataset.criteria.data[:nb_row_set, :, :, -1] == Criteria.VALID)


class TestMaskBorder:
    """Test mask_border method."""

    def test_null_offset(self, cost_volumes, criteria_dataset):
        """Window_size == 1 -> offset = 0, no raise PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria"""
        make_criteria_copy = criteria_dataset.copy(deep=True)
        criteria.mask_border(cost_volumes, criteria_dataset)

        # Check criteria_dataset has not changed
        xr.testing.assert_equal(criteria_dataset, make_criteria_copy)
        # Check the PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria does not raise
        assert np.all(criteria_dataset.criteria.data[:, :, :, :] != Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)

    @pytest.mark.parametrize("window_size", [3, 5, 7])
    def test_variable_offset(self, cost_volumes, criteria_dataset):
        """
        Window_size == X -> offset = int((window_size - 1) / 2)
        With mask_border, the PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria is raised on the border.

        Example :
        window_size = 3 -> offset = 1

        For this image :          1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8

        and a criteria_dataset :  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0

        the result is :           1 1 1 1 1 1 1 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 1 1 1 1 1 1 1
        """
        criteria.mask_border(cost_volumes, criteria_dataset)

        offset = cost_volumes.offset_row_col
        assert np.all(criteria_dataset.criteria.data[:offset, :, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataset.criteria.data[-offset:, :, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataset.criteria.data[:, :offset, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataset.criteria.data[:, -offset:, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)


class TestMaskDisparityOutsideRightImage:
    """Test mask_disparity_outside_right_image method."""

    @pytest.fixture()
    def offset(self, cost_volumes):
        return cost_volumes.offset_row_col

    @pytest.fixture()
    def ground_truth_null_disparity(self, offset, img_size):
        """Make ground_truth of criteria dataset for null disparity"""
        data = np.full(img_size, Criteria.VALID)
        if offset > 0:
            data[:offset, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[-offset:, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[:, :offset] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[:, -offset:] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        return data

    @pytest.fixture()
    def ground_truth_first_disparity(self, cost_volumes, offset, img_size):
        """
        Make ground_truth of criteria dataset for first disparity (disp_col=-5 and disp_row=-1)

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
        delta_row_start = offset + abs(cost_volumes.disp_row.values[0])
        delta_row_end = offset + cost_volumes.disp_row.values[0]
        data[:delta_row_start, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        if delta_row_end > 0:
            data[-delta_row_end:, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        # Udpate col
        delta_col_start = offset + abs(cost_volumes.disp_col.values[0])
        delta_col_end = offset + cost_volumes.disp_col.values[0]
        data[:, :delta_col_start] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        if delta_col_end > 0:
            data[:, -delta_col_end:] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        return data

    @pytest.mark.parametrize(
        "window_size",
        [
            pytest.param(1, id="offset nul"),
            pytest.param(3, id="offset == 1"),
            pytest.param(5, id="offset == 2"),
            pytest.param(7, id="offset == 3"),
            pytest.param(99, id="offset > dimension"),
        ],
    )
    def test_nominal(self, cost_volumes, criteria_dataset, ground_truth_null_disparity, ground_truth_first_disparity):
        """
        Test mask_disparity_outside_right_image
        """
        criteria.mask_disparity_outside_right_image(cost_volumes, criteria_dataset)

        np.testing.assert_array_equal(criteria_dataset.criteria.values[:, :, 5, 1], ground_truth_null_disparity)
        np.testing.assert_array_equal(criteria_dataset.criteria.values[:, :, 0, 0], ground_truth_first_disparity)
