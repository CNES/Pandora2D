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
Test methods from criteria.py file
"""
# pylint: disable=too-many-lines
# pylint: disable=redefined-outer-name

import copy
import pytest
import numpy as np
import xarray as xr

from pandora2d import matching_cost, criteria
from pandora2d.constants import Criteria
from pandora2d.img_tools import add_disparity_grid, create_datasets_from_inputs
from pandora2d.state_machine import Pandora2DMachine
from pandora2d import run
from pandora2d.check_configuration import check_conf


@pytest.fixture()
def img_size():
    row = 10
    col = 13
    return (row, col)


@pytest.fixture()
def disparity_cfg():
    """Return (disp_row, disp_col)"""
    return {"init": 1, "range": 2}, {"init": -1, "range": 4}


@pytest.fixture()
def no_data_mask():
    return 1


@pytest.fixture()
def valid_pixels():
    return 0


@pytest.fixture()
def image(img_size, disparity_cfg, valid_pixels, no_data_mask):
    """Make image"""
    row, col = img_size
    row_disparity, col_disparity = disparity_cfg
    data = np.random.uniform(0, row * col, (row, col))

    return xr.Dataset(
        {
            "im": (["row", "col"], data),
            "msk": (["row", "col"], np.zeros_like(data)),
        },
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        attrs={
            "no_data_img": -9999,
            "valid_pixels": valid_pixels,
            "no_data_mask": no_data_mask,
            "crs": None,
            "invalid_disparity": np.nan,
        },
    ).pipe(add_disparity_grid, col_disparity, row_disparity)


@pytest.fixture()
def mask_image(image, msk):
    image["msk"].data = msk


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
def cost_volumes(matching_cost_cfg, image):
    """Compute a cost_volumes"""
    matching_cost_ = matching_cost.PandoraMatchingCostMethods(matching_cost_cfg)

    matching_cost_.allocate(img_left=image, img_right=image, cfg=matching_cost_cfg)
    return matching_cost_.compute_cost_volumes(img_left=image, img_right=image)


@pytest.fixture()
def criteria_dataarray(img_size):
    shape = (*img_size, 5, 9)
    return xr.DataArray(
        np.full(shape, Criteria.VALID),
        coords={
            "row": np.arange(shape[0]),
            "col": np.arange(shape[1]),
            "disp_row": np.arange(-1, 4),
            "disp_col": np.arange(-5, 4),
        },
        dims=["row", "col", "disp_row", "disp_col"],
    )


class TestCriteria:
    """Test the Criteria Enum."""

    def test_can_be_stored_in_uint8_np_array(self):
        """Criteria can be stored in uint8 numpy array."""
        result = np.array([Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER], dtype=np.uint8)
        assert result.dtype == np.uint8

    def test_is_in(self):
        """Test method to see if a Criteria is part of an array."""
        data = np.array(
            [
                Criteria.VALID,
                Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER,
                Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER | Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE,
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER.is_in(data), [False, True, True])
        np.testing.assert_array_equal(Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE.is_in(data), [False, False, True])


class TestFlagArray:
    """Test flag array."""

    @pytest.fixture(scope="class")
    def flag_array(self):
        return criteria.FlagArray(
            [
                Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE,
                Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA,
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
            f"{prefix}([<PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE: "
            f"{Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE.value}>,\n{prefix_offset}"
            f"<PANDORA2D_MSK_PIXEL_RIGHT_NODATA: "
            f"{Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA.value}>], "
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


class TestSetUnprocessedDisparity:
    """Test create a criteria xarray.Dataset."""

    @pytest.fixture()
    def grid_min_col(self, image):
        return image["col_disparity"].sel(band_disp="min")

    @pytest.fixture()
    def grid_max_col(self, image):
        return image["col_disparity"].sel(band_disp="max")

    @pytest.fixture()
    def grid_min_row(self, image):
        return image["row_disparity"].sel(band_disp="min")

    @pytest.fixture()
    def grid_max_row(self, image):
        return image["row_disparity"].sel(band_disp="max")

    def test_homogeneous_grids(self, criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row):
        """With uniform grids"""
        make_criteria_copy = criteria_dataarray.copy(deep=True)
        criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        xr.testing.assert_equal(criteria_dataarray, make_criteria_copy)

    def test_variable_col_disparity(
        self, criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size
    ):
        """With variable column disparity grids"""
        _, col = img_size
        nb_col_set = int(col / 2)
        grid_min_col[:, :nb_col_set] = criteria_dataarray.coords["disp_col"].data[1]
        grid_max_col[:, nb_col_set:] = criteria_dataarray.coords["disp_col"].data[-2]

        criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        assert np.all(
            criteria_dataarray.data[:, :nb_col_set, :, 0] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataarray.data[:, nb_col_set:, :, 0] == Criteria.VALID)
        assert np.all(
            criteria_dataarray.data[:, nb_col_set:, :, -1] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataarray.data[:, :nb_col_set, :, -1] == Criteria.VALID)

    def test_variable_row_disparity(
        self, criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size
    ):
        """With variable row disparity grids"""
        row, _ = img_size
        nb_row_set = int(row / 2)
        grid_min_row[:nb_row_set, :] = criteria_dataarray.coords["disp_row"].data[1]
        grid_max_row[nb_row_set:, :] = criteria_dataarray.coords["disp_row"].data[-2]

        criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

        assert np.all(
            criteria_dataarray.data[:nb_row_set, :, 0, :] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataarray.data[nb_row_set:, :, 0, :] == Criteria.VALID)
        assert np.all(
            criteria_dataarray.data[nb_row_set:, :, -1, :] == Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
        )
        assert np.all(criteria_dataarray.data[:nb_row_set, :, -1, :] == Criteria.VALID)


class TestMaskBorder:
    """Test mask_border method."""

    def test_null_offset(self, criteria_dataarray):
        """offset = 0, no raise PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria"""
        make_criteria_copy = criteria_dataarray.copy(deep=True)
        criteria.mask_border(0, criteria_dataarray)

        # Check criteria_dataarray has not changed
        xr.testing.assert_equal(criteria_dataarray, make_criteria_copy)
        # Check the PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria does not raise
        assert np.all(criteria_dataarray.data[:, :, :, :] != Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)

    @pytest.mark.parametrize("offset", [1, 2, 3])
    def test_variable_offset(self, criteria_dataarray, offset):
        """
        With mask_border, the PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria is raised on the border.

        Example :
        offset = 1

        For this image :          1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8

        and a criteria_dataarray :  0 0 0 0 0 0 0 0
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
        criteria.mask_border(offset, criteria_dataarray)

        assert np.all(criteria_dataarray.data[:offset, :, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataarray.data[-offset:, :, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataarray.data[:, :offset, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)
        assert np.all(criteria_dataarray.data[:, -offset:, :, :] == Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER)


class TestMaskDisparityOutsideRightImage:
    """Test mask_disparity_outside_right_image method."""

    @pytest.fixture()
    def ground_truth_null_disparity(self, offset, img_size):
        """Make ground_truth of criteria dataarray for null disparity"""
        data = np.full(img_size, Criteria.VALID)
        if offset > 0:
            data[:offset, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[-offset:, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[:, :offset] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
            data[:, -offset:] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        return data

    @pytest.fixture()
    def ground_truth_first_disparity(self, offset, img_size):
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
        data[:delta_row_start, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        if delta_row_end > 0:
            data[-delta_row_end:, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        # Udpate col
        first_col_disparity = -5
        delta_col_start = offset + abs(first_col_disparity)
        delta_col_end = offset + first_col_disparity
        data[:, :delta_col_start] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        if delta_col_end > 0:
            data[:, -delta_col_end:] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
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
    def test_nominal(self, offset, criteria_dataarray, ground_truth_null_disparity, ground_truth_first_disparity):
        """
        Test mask_disparity_outside_right_image
        """
        criteria.mask_disparity_outside_right_image(offset, criteria_dataarray)

        np.testing.assert_array_equal(criteria_dataarray.values[:, :, 1, 5], ground_truth_null_disparity)
        np.testing.assert_array_equal(criteria_dataarray.values[:, :, 0, 0], ground_truth_first_disparity)


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
        expected_criteria_data[row_slice, col_slice, ...] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA

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

        criteria_dataarray.data[no_data_row_position, no_data_col_position, ...] = (
            Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        )

        expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
        expected_criteria_data[row_slice, col_slice, ...] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA
        expected_criteria_data[no_data_row_position, no_data_col_position, ...] = (
            Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        )

        criteria.mask_left_no_data(image, window_size, criteria_dataarray)

        np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)


@pytest.mark.parametrize("img_size", [(4, 5)])
class TestMaskRightNoData:
    """Test mask_right_no_data function."""

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["no_data_mask", "msk", "disp_row", "disp_col", "expected_criteria"],
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
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
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
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
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
                np.array(
                    # fmt: off
                    [
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                    ]
                    # fmt: on
                ),
                id="Disp 2 1 - other no_data_mask",
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
        ["no_data_mask", "msk", "disp_row", "disp_col", "expected_criteria"],
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
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA],
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
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
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
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
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
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
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
                np.array(
                    [
                        # fmt: off
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                id="Disp 2 1 - Pos (2,3)",
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

        criteria_dataarray.data[2, 3, ...] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

        criteria.mask_right_no_data(image, 1, criteria_dataarray)

        assert (
            criteria_dataarray.sel(row=2, col=3, disp_row=1, disp_col=1).data
            == Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA
        )


class TestMaskLeftInvalid:
    """Test mask_left_invalid function."""

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
    def test_mask_left_invalid(self, img_size, image, criteria_dataarray, invalid_position):
        """
        Test that mask_invalid_left method raises criteria PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT
        for points whose value is neither valid_pixels or no_data_mask.
        """
        invalid_row_position, invalid_col_position = invalid_position

        # We put 2 in img_left msk because it is different from valid_pixels=0 and no_data_mask=1
        image["msk"][invalid_row_position, invalid_col_position] = 2

        expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
        expected_criteria_data[invalid_row_position, invalid_col_position, ...] = (
            Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT
        )

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
    def test_add_to_existing(self, img_size, image, criteria_dataarray, invalid_position):
        """Test we do not override existing criteria but combine it."""
        invalid_row_position, invalid_col_position = invalid_position

        # We put 2 in img_left msk because it is different from valid_pixels=0 and no_data_mask=1
        image["msk"][invalid_row_position, invalid_col_position] = 2

        criteria_dataarray.data[invalid_row_position, invalid_col_position, ...] = (
            Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        )

        expected_criteria_data = np.full((*img_size, 5, 9), Criteria.VALID)
        expected_criteria_data[invalid_row_position, invalid_col_position, ...] = (
            Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
        )

        criteria.mask_left_invalid(image, criteria_dataarray)

        np.testing.assert_array_equal(criteria_dataarray.values, expected_criteria_data)


@pytest.mark.parametrize("img_size", [(4, 5)])
class TestMaskRightInvalid:
    """Test mask_right_invalid function."""

    @pytest.mark.usefixtures("mask_image")
    @pytest.mark.parametrize(
        ["valid_pixels", "no_data_mask", "msk", "expected_criteria", "disp_col", "disp_row"],
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
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT],
                        # fmt: on
                    ]
                ),
                -1,  # disp_col
                -1,  # disp_row
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
                        [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                2,  # disp_col
                1,  # disp_row
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
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                -1,  # disp_col
                -1,  # disp_row
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
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,  # disp_col
                -1,  # disp_row
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
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        # fmt: on
                    ]
                ),
                0,  # disp_col
                -1,  # disp_row
                id="Invalid point at center of right mask with no_data_mask=4, valid_pixels=3, disp_row=-1 and disp_col=0",
            ),
            # pylint: enable=line-too-long
        ],
    )
    def test_mask_invalid_right(self, image, criteria_dataarray, expected_criteria, disp_col, disp_row):
        """
        Test that mask_invalid_right method raises criteria PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT
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
        Test that we combine Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT
        with existing criteria and do not override them.
        """

        criteria_dataarray.data[2, 3, ...] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

        criteria.mask_right_invalid(image, criteria_dataarray)

        assert (
            criteria_dataarray.sel(row=2, col=3, disp_row=disp_row, disp_col=disp_col).data
            == Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE | Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT
        )


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
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
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
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                2,  # disp_col
                -1,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
                        [Criteria.VALID , Criteria.VALID , Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
                        [Criteria.VALID , Criteria.VALID , Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
                        # fmt: on
                    ]
                ),
                id="Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED overcome other criteria",
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
                        [Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.VALID, Criteria.VALID, Criteria.VALID, Criteria.VALID],
                        [Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA],
                        [Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT, Criteria.VALID],
                        [Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE , Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE ],
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
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE | Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER , Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
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
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA  | Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT, Criteria.VALID, Criteria.VALID],
                        [Criteria.VALID, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT, Criteria.VALID, Criteria.VALID],
                        [Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE],
                        # fmt: on
                    ]
                ),
                id="Centered invalid and no data in msk with window_size=1",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
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
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
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
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        [Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER, Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER],
                        # fmt: on
                    ]
                ),
                id="Window_size=5, only Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER is raised",
            ),
            pytest.param(
                np.array(  # left msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.array(  # right msk
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                -5,  # disp_col
                0,  # disp_row
                1,  # window_size
                np.array(
                    [
                        # fmt: off
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE],
                        [Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE, Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE],
                        # fmt: on
                    ]
                ),
                id="Column disparity out of the image or unprocessed for all points",
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


@pytest.fixture()
def ground_truth_criteria_dataarray(left_img_shape):
    """
    Criteria dataarray ground truth
    for test_criteria_datarray_created_in_state_machine.

    This ground truth is based on the parameters (window_size, disparity) of the
    correct_input_cfg and correct_pipeline_without_refinement fixtures.
    """

    # disp = {"init": 1, "range":2} -> range size = 5
    ground_truth = np.full((left_img_shape[0], left_img_shape[1], 5, 5), Criteria.VALID)

    # Here, window_size=5
    # For disp=-1, 3 first column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:3, :, 0, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, :3, :, 0] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=1, 3 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-3:, :, 2, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -3:, :, 2] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=2, 4 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-4:, :, 3, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -4:, :, 3] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=3, 5 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-5:, :, 4, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -5:, :, 4] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # Window_size=5, so the two first and last rows and columns are equal to Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:2, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:, :2, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[-2:, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:, -2:, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER

    return ground_truth


def test_criteria_datarray_created_in_state_machine(
    correct_input_cfg, correct_pipeline_without_refinement, ground_truth_criteria_dataarray
):
    """
    Test that pandora2d machine contains the criteria dataarray
    """

    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement, "output": {"path": "rala"}}

    img_left, img_right = create_datasets_from_inputs(input_config=correct_input_cfg["input"])

    pandora2d_machine = Pandora2DMachine()

    checked_cfg = check_conf(configuration, pandora2d_machine)

    dataset_disp_maps, _ = run(pandora2d_machine, img_left, img_right, checked_cfg)

    # Get peak on the edges to add Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE in ground_truth_criteria_dataarray
    row_peak_mask = (
        dataset_disp_maps["row_map"].data
        == correct_input_cfg["input"]["row_disparity"]["init"] - correct_input_cfg["input"]["row_disparity"]["range"]
    ) | (
        dataset_disp_maps["row_map"].data
        == correct_input_cfg["input"]["row_disparity"]["init"] + correct_input_cfg["input"]["row_disparity"]["range"]
    )

    col_peak_mask = (
        dataset_disp_maps["col_map"].data
        == correct_input_cfg["input"]["col_disparity"]["init"] - correct_input_cfg["input"]["col_disparity"]["range"]
    ) | (
        dataset_disp_maps["col_map"].data
        == correct_input_cfg["input"]["col_disparity"]["init"] + correct_input_cfg["input"]["col_disparity"]["range"]
    )

    ground_truth_criteria_dataarray[row_peak_mask | col_peak_mask] |= Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE

    # Check that criteria_dataarray exists and is the same shape as the cost_volumes object
    assert pandora2d_machine.cost_volumes.cost_volumes.sizes == pandora2d_machine.criteria_dataarray.sizes
    # Check that criteria dataarray contains correct criteria
    np.testing.assert_array_equal(pandora2d_machine.criteria_dataarray.data, ground_truth_criteria_dataarray)


class TestPeakOnEdge:
    """
    Test the methods linked to PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE criteria
    """

    @pytest.fixture()
    def row_map(self, img_size, disparity_cfg):
        """
        Row disparity map used for tests
        """

        row_map = np.full(img_size, 2)

        # row_map[0,0] is equal to the minimum of the row disparity range
        row_map[0, 0] = disparity_cfg[0]["init"] - disparity_cfg[0]["range"]
        # row_map[3,3] is equal to the maximum of the row disparity range
        row_map[3, 3] = disparity_cfg[0]["init"] + disparity_cfg[0]["range"]
        return row_map

    @pytest.fixture()
    def col_map(self, img_size, disparity_cfg):
        """
        Col disparity map used for tests
        """

        col_map = np.full(img_size, -1)

        # col_map[0,0] is equal to the maximum of the col disparity range
        col_map[0, 0] = disparity_cfg[1]["init"] + disparity_cfg[1]["range"]
        # col_map[0,0] is equal to the minimum of the col disparity range
        col_map[4, 5] = disparity_cfg[1]["init"] - disparity_cfg[1]["range"]
        return col_map

    @pytest.fixture()
    def row_map_full_peak(self, img_size, disparity_cfg):
        """
        Row disparity map with only peak on edges used for tests
        """

        # row_map is filled with the minimum of the row disparity range
        row_map = np.full(img_size, disparity_cfg[0]["init"] - disparity_cfg[0]["range"])
        return row_map

    @pytest.fixture()
    def col_map_full_peak(self, img_size, disparity_cfg):
        """
        Col disparity map with only peak on edges used for tests
        """

        # col_map is filled with the maximum of the col disparity range
        col_map = np.full(img_size, disparity_cfg[1]["init"] + disparity_cfg[1]["range"])
        return col_map

    @pytest.fixture()
    def map_without_peak(self, img_size):
        """
        Disparity map without peak on edges
        """

        return np.full(img_size, 1)

    def test_apply_peak_on_edge(self, criteria_dataarray, image, cost_volumes, row_map, col_map):
        """
        Test the apply_peak_on_edge method
        """

        cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

        criteria.apply_peak_on_edge(criteria_dataarray, image, cost_volumes_coords, row_map, col_map)

        assert (criteria_dataarray.data[0, 0, :, :] == Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE).all()
        assert (criteria_dataarray.data[4, 5, :, :] == Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE).all()
        assert (criteria_dataarray.data[3, 3, :, :] == Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE).all()

    @pytest.mark.parametrize(
        ["drow_map", "dcol_map"],
        [
            pytest.param("row_map_full_peak", "col_map_full_peak", id="Row and col disparity maps full of peaks"),
            pytest.param("row_map_full_peak", "col_map", id="Row map full of peaks"),
            pytest.param("map_without_peak", "col_map_full_peak", id="Col map full of peaks"),
        ],
    )
    def test_apply_peak_on_edge_full_peak_map(
        self, criteria_dataarray, image, cost_volumes, drow_map, dcol_map, request
    ):
        """
        Test the apply_peak_on_edge method with disparity maps full of peaks on edges
        """

        cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

        criteria.apply_peak_on_edge(
            criteria_dataarray,
            image,
            cost_volumes_coords,
            request.getfixturevalue(drow_map),
            request.getfixturevalue(dcol_map),
        )

        assert (criteria_dataarray.data == Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE).all()

    def test_apply_peak_on_edge_without_peak(self, criteria_dataarray, image, cost_volumes, map_without_peak):
        """
        Test the apply_peak_on_edge method with maps without peaks on edges
        """

        cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

        criteria.apply_peak_on_edge(criteria_dataarray, image, cost_volumes_coords, map_without_peak, map_without_peak)

        assert (criteria_dataarray.data != Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE).all()
