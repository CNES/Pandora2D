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
Test image shift methods
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import pytest
import xarray as xr
import numpy as np


from pandora2d import img_tools


@pytest.fixture()
def no_data_img_attribute():
    """No data image"""
    return -9999


@pytest.fixture()
def monoband_image(no_data_img_attribute):
    """Create monoband image."""
    data = np.array(
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        dtype=np.float32,
    )

    return xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
    ).assign_attrs(
        {
            "no_data_img": no_data_img_attribute,
        }
    )


@pytest.fixture()
def roi_image(monoband_image):
    """Create ROI image."""
    return monoband_image.assign_coords({"row": np.arange(2, 7), "col": np.arange(5, 11)})


@pytest.fixture()
def no_data_image(no_data_img_attribute):
    """Create an image with no_data=-9999"""

    data = np.array(
        (
            [1, 1, 1, 1, -9999, 1],
            [-9999, 1, 1, 1, 2, 1],
            [1, 1, 1, 4, 3, 1],
            [1, 1, 1, -9999, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ),
        dtype=np.float32,
    )

    return xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(2, 7), "col": np.arange(5, 11)}
    ).assign_attrs(
        {
            "no_data_img": no_data_img_attribute,
        }
    )


class TestShiftDispRowImg:
    """
    Test shift_disp_row_img method
    """

    @pytest.mark.parametrize(
        ["image", "disp_row", "no_data_img_attribute", "expected"],
        [
            pytest.param(
                "monoband_image",
                1,
                -9999,
                np.array(
                    (
                        [1, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                    ),
                    dtype=np.float32,
                ),
                id="monoband image, disp_row=1 and no_data_img=-9999",
            ),
            pytest.param(
                "monoband_image",
                1,
                5,
                np.array(
                    (
                        [1, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [5, 5, 5, 5, 5, 5],
                    ),
                    dtype=np.float32,
                ),
                id="monoband image, disp_row=1 and no_data_img=5",
            ),
            pytest.param(
                "monoband_image",
                -2,
                -9999,
                np.array(
                    (
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                    ),
                    dtype=np.float32,
                ),
                id="monoband image, disp_row=-2 and no_data_img=-9999",
            ),
            pytest.param(
                "no_data_image",
                1,
                -9999,
                np.array(
                    (
                        [-9999, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                        [1, 1, 1, -9999, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                    ),
                    dtype=np.float32,
                ),
                id="no_data image, disp_row=1 and no_data_img=-9999",
            ),
        ],
    )
    def test_shift_disp_row_img(self, image, disp_row, no_data_img_attribute, expected, request):
        """
        Test shift_disp_row_img method
        """

        image_to_shift = request.getfixturevalue(image)
        shifted_img = img_tools.shift_disp_row_img(image_to_shift, disp_row)

        np.testing.assert_array_equal(shifted_img["im"].data, expected)
        np.testing.assert_array_equal(shifted_img.row.values, image_to_shift.row.values)
        np.testing.assert_array_equal(shifted_img.col.values, image_to_shift.col.values)
        assert shifted_img.attrs["no_data_img"] == no_data_img_attribute

    @pytest.mark.parametrize(
        ["image", "disp_row", "expected"],
        [
            pytest.param(
                "monoband_image",
                1,
                np.array(
                    (
                        [1, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                    ),
                    dtype=np.float32,
                ),
                id="monoband image disp_row=1",
            ),
            pytest.param(
                "monoband_image",
                -2,
                np.array(
                    (
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999, -9999, -9999],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 2, 1],
                        [1, 1, 1, 4, 3, 1],
                    ),
                    dtype=np.float32,
                ),
                id="monoband image disp_row=-2",
            ),
        ],
    )
    def test_shift_disp_row_img_with_no_data_is_nan(self, image, disp_row, expected, request):
        """
        Test shift_disp_row_img method when right image is given with "NaN" as no data value.
        We check that the value of no_data has been replaced by -9999 and that there is no propagation of NaN.
        """

        image_to_shift = request.getfixturevalue(image)
        image_to_shift.attrs["no_data_img"] = np.nan
        shifted_img = img_tools.shift_disp_row_img(image_to_shift, disp_row)

        np.testing.assert_array_equal(shifted_img["im"].data, expected)
        np.testing.assert_array_equal(shifted_img.row.values, image_to_shift.row.values)
        np.testing.assert_array_equal(shifted_img.col.values, image_to_shift.col.values)
        assert shifted_img.attrs["no_data_img"] == -9999


class TestShiftSubpixImg:
    """Test shift_subpix_img function."""

    @pytest.mark.parametrize(
        ["image", "subpix", "number", "expected"],
        [
            pytest.param(
                "monoband_image", 4, 1, np.array([0.25, 1.25, 2.25, 3.25, 4.25, 5.25]), id="monoband image subpix 0.25"
            ),
            pytest.param(
                "monoband_image", 4, 2, np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), id="monoband image subpix 0.5"
            ),
            pytest.param(
                "monoband_image", 4, 3, np.array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75]), id="monoband image subpix 0.75"
            ),
            pytest.param("roi_image", 2, 1, np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]), id="roi image subpix 0.25"),
        ],
    )
    def test_column(self, image, subpix, number, expected, request):
        """
        Test shift_subpix_img function for column shift
        """
        shifted_img = img_tools.shift_subpix_img(request.getfixturevalue(image), subpix, False)

        # check if columns coordinates has been shifted
        np.testing.assert_array_equal(expected, shifted_img[number].col)

    @pytest.mark.parametrize(
        ["image", "subpix", "number", "expected"],
        [
            pytest.param(
                "monoband_image", 4, 1, np.array([0.25, 1.25, 2.25, 3.25, 4.25]), id="monoband image subpix 0.25"
            ),
            pytest.param("monoband_image", 4, 2, np.array([0.5, 1.5, 2.5, 3.5, 4.5]), id="monoband image subpix 0.5"),
            pytest.param(
                "monoband_image", 4, 3, np.array([0.75, 1.75, 2.75, 3.75, 4.75]), id="monoband image subpix 0.75"
            ),
            pytest.param("roi_image", 2, 1, np.array([2.5, 3.5, 4.5, 5.5, 6.5]), id="monoband image subpix 0.5"),
        ],
    )
    def test_row(self, image, subpix, number, expected, request):
        """
        Test shift_subpix_img function for row shift
        """
        shifted_img = img_tools.shift_subpix_img(request.getfixturevalue(image), subpix, True)

        # check if row coordinates has been shifted
        np.testing.assert_array_equal(expected, shifted_img[number].row)

    def test_apply_row_and_col(self, monoband_image):
        """
        Test shift_subpix_img function for row and col shift or col and row shift
        """

        shifted_img = img_tools.shift_subpix_img(monoband_image, 2, True)
        shifted_img_row_and_col = img_tools.shift_subpix_img(shifted_img[1], 2, False)

        shifted_img = img_tools.shift_subpix_img(monoband_image, 2, False)
        shifted_img_col_and_row = img_tools.shift_subpix_img(shifted_img[1], 2, True)

        # check if data is the same
        np.testing.assert_array_equal(shifted_img_row_and_col[1]["im"].data, shifted_img_col_and_row[1]["im"].data)

    def test_difference_between_row_and_col(self, monoband_image):
        """
        Test shift_subpix_img function for row shift or col shift
        """
        shifted_row_img = img_tools.shift_subpix_img(monoband_image, 2, True)

        shifted_col_img = img_tools.shift_subpix_img(monoband_image, 2, False)

        # Test that the last row of shifted_row_img is full of no data
        np.testing.assert_array_equal(
            shifted_row_img[1]["im"][-1, :],
            np.full((len(shifted_row_img[1]["im"][-1, :])), monoband_image.attrs["no_data_img"]),
        )

        # Test that the last column of shifted_col_img is full of no data
        np.testing.assert_array_equal(
            shifted_col_img[1]["im"][:, -1],
            np.full((len(shifted_col_img[1]["im"][:, -1])), monoband_image.attrs["no_data_img"]),
        )

        # Test that the content of shifted_row_img and shifted_col_img are different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(shifted_row_img[1]["im"], shifted_col_img[1]["im"])


class TestShiftSubpixImg2d:
    """Test shift_subpix_img_2d function."""

    # pylint:disable=too-few-public-methods

    @pytest.mark.parametrize(
        ["image", "subpix", "number", "expected_row", "expected_col"],
        [
            pytest.param(
                "monoband_image",
                1,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=1",
            ),
            pytest.param(
                "monoband_image",
                2,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=2, index=0",
            ),
            pytest.param(
                "monoband_image",
                2,
                1,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=2, index=1",
            ),
            pytest.param(
                "monoband_image",
                2,
                2,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=2, index=2",
            ),
            pytest.param(
                "monoband_image",
                2,
                3,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=2, index=3",
            ),
            pytest.param(
                "monoband_image",
                4,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=4, index=0",
            ),
            pytest.param(
                "monoband_image",
                4,
                1,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.25, 1.25, 2.25, 3.25, 4.25, 5.25]),
                id="monoband image subpix=4, index=1",
            ),
            pytest.param(
                "monoband_image",
                4,
                2,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=4, index=2",
            ),
            pytest.param(
                "monoband_image",
                4,
                3,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75]),
                id="monoband image subpix=4, index=3",
            ),
            pytest.param(
                "monoband_image",
                4,
                4,
                np.array([0.25, 1.25, 2.25, 3.25, 4.25]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=4, index=4",
            ),
            pytest.param(
                "monoband_image",
                4,
                5,
                np.array([0.25, 1.25, 2.25, 3.25, 4.25]),
                np.array([0.25, 1.25, 2.25, 3.25, 4.25, 5.25]),
                id="monoband image subpix=4, index=5",
            ),
            pytest.param(
                "monoband_image",
                4,
                6,
                np.array([0.25, 1.25, 2.25, 3.25, 4.25]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=4, index=6",
            ),
            pytest.param(
                "monoband_image",
                4,
                7,
                np.array([0.25, 1.25, 2.25, 3.25, 4.25]),
                np.array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75]),
                id="monoband image subpix=4, index=7",
            ),
            pytest.param(
                "monoband_image",
                4,
                8,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=4, index=8",
            ),
            pytest.param(
                "monoband_image",
                4,
                9,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.25, 1.25, 2.25, 3.25, 4.25, 5.25]),
                id="monoband image subpix=4, index=9",
            ),
            pytest.param(
                "monoband_image",
                4,
                10,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=4, index=10",
            ),
            pytest.param(
                "monoband_image",
                4,
                11,
                np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                np.array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75]),
                id="monoband image subpix=4, index=11",
            ),
            pytest.param(
                "monoband_image",
                4,
                12,
                np.array([0.75, 1.75, 2.75, 3.75, 4.75]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                id="monoband image subpix=4, index=12",
            ),
            pytest.param(
                "monoband_image",
                4,
                13,
                np.array([0.75, 1.75, 2.75, 3.75, 4.75]),
                np.array([0.25, 1.25, 2.25, 3.25, 4.25, 5.25]),
                id="monoband image subpix=4, index=13",
            ),
            pytest.param(
                "monoband_image",
                4,
                14,
                np.array([0.75, 1.75, 2.75, 3.75, 4.75]),
                np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
                id="monoband image subpix=4, index=14",
            ),
            pytest.param(
                "monoband_image",
                4,
                15,
                np.array([0.75, 1.75, 2.75, 3.75, 4.75]),
                np.array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75]),
                id="monoband image subpix=4, index=15",
            ),
            pytest.param(
                "roi_image",
                1,
                0,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=1",
            ),
            pytest.param(
                "roi_image",
                2,
                0,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=2, index=0",
            ),
            pytest.param(
                "roi_image",
                2,
                1,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=2, index=1",
            ),
            pytest.param(
                "roi_image",
                2,
                2,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=2, index=2",
            ),
            pytest.param(
                "roi_image",
                2,
                3,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=2, index=3",
            ),
            pytest.param(
                "roi_image",
                4,
                0,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=4, index=0",
            ),
            pytest.param(
                "roi_image",
                4,
                1,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.25, 6.25, 7.25, 8.25, 9.25, 10.25]),
                id="roi image subpix=4, index=1",
            ),
            pytest.param(
                "roi_image",
                4,
                2,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=4, index=2",
            ),
            pytest.param(
                "roi_image",
                4,
                3,
                np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([5.75, 6.75, 7.75, 8.75, 9.75, 10.75]),
                id="roi image subpix=4, index=3",
            ),
            pytest.param(
                "roi_image",
                4,
                4,
                np.array([2.25, 3.25, 4.25, 5.25, 6.25]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=4, index=4",
            ),
            pytest.param(
                "roi_image",
                4,
                5,
                np.array([2.25, 3.25, 4.25, 5.25, 6.25]),
                np.array([5.25, 6.25, 7.25, 8.25, 9.25, 10.25]),
                id="roi image subpix=4, index=5",
            ),
            pytest.param(
                "roi_image",
                4,
                6,
                np.array([2.25, 3.25, 4.25, 5.25, 6.25]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=4, index=6",
            ),
            pytest.param(
                "roi_image",
                4,
                7,
                np.array([2.25, 3.25, 4.25, 5.25, 6.25]),
                np.array([5.75, 6.75, 7.75, 8.75, 9.75, 10.75]),
                id="roi image subpix=4, index=7",
            ),
            pytest.param(
                "roi_image",
                4,
                8,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=4, index=8",
            ),
            pytest.param(
                "roi_image",
                4,
                9,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.25, 6.25, 7.25, 8.25, 9.25, 10.25]),
                id="roi image subpix=4, index=9",
            ),
            pytest.param(
                "roi_image",
                4,
                10,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=4, index=10",
            ),
            pytest.param(
                "roi_image",
                4,
                11,
                np.array([2.5, 3.5, 4.5, 5.5, 6.5]),
                np.array([5.75, 6.75, 7.75, 8.75, 9.75, 10.75]),
                id="roi image subpix=4, index=11",
            ),
            pytest.param(
                "roi_image",
                4,
                12,
                np.array([2.75, 3.75, 4.75, 5.75, 6.75]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                id="roi image subpix=4, index=12",
            ),
            pytest.param(
                "roi_image",
                4,
                13,
                np.array([2.75, 3.75, 4.75, 5.75, 6.75]),
                np.array([5.25, 6.25, 7.25, 8.25, 9.25, 10.25]),
                id="roi image subpix=4, index=13",
            ),
            pytest.param(
                "roi_image",
                4,
                14,
                np.array([2.75, 3.75, 4.75, 5.75, 6.75]),
                np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
                id="roi image subpix=4, index=14",
            ),
            pytest.param(
                "roi_image",
                4,
                15,
                np.array([2.75, 3.75, 4.75, 5.75, 6.75]),
                np.array([5.75, 6.75, 7.75, 8.75, 9.75, 10.75]),
                id="roi image subpix=4, index=15",
            ),
        ],
    )
    def test_shift_subpix_img_2d(self, image, subpix, number, expected_row, expected_col, request):
        """
        Test the shift_subpix_img_2d method
        """

        shifted_img_2d = img_tools.shift_subpix_img_2d(request.getfixturevalue(image), subpix)

        assert len(shifted_img_2d) == subpix**2
        np.testing.assert_array_equal(expected_row, shifted_img_2d[number].row)
        np.testing.assert_array_equal(expected_col, shifted_img_2d[number].col)
