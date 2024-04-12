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
Test configuration
"""
import unittest
import pytest
import xarray as xr
import numpy as np


from pandora2d import img_tools


class TestShiftDispRowImg(unittest.TestCase):
    """
    test shift_disp_row_img function.
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """
        # original image
        data = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))
        # original mask
        mask = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        # create original dataset
        self.data = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        # add attributes for mask
        self.data.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,  # arbitrary default value
            "no_data_mask": 1,
        }
        no_data_pixels = np.where(data == np.nan)
        self.data["msk"] = xr.DataArray(
            np.full((data.shape[0], data.shape[1]), self.data.attrs["valid_pixels"]).astype(np.int16),
            dims=["row", "col"],
        )
        # associate nan value in mask to the no_data param
        self.data["msk"].data[no_data_pixels] = int(self.data.attrs["no_data_mask"])

        # create the dataset of an image with dec_y = 1
        shifted_data = np.array([[1, 1, 1], [1, 1, 1], [-9999, -9999, -9999]])
        # original mask
        shifted_mask = np.array(([1, 1, 1], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        self.data_down = xr.Dataset(
            {"im": (["row", "col"], shifted_data), "msk": (["row", "col"], shifted_mask)},
            coords={"row": np.arange(shifted_data.shape[0]), "col": np.arange(shifted_data.shape[1])},
        )

        self.data_down.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,  # arbitrary default value
            "no_data_mask": 1,
        }

        no_data_pixels = np.where(shifted_data == -9999)
        self.data_down["msk"] = xr.DataArray(
            np.full((shifted_data.shape[0], shifted_data.shape[1]), self.data_down.attrs["valid_pixels"]).astype(
                np.int16
            ),
            dims=["row", "col"],
        )
        # associate nan value in mask to the no_data param
        self.data_down["msk"].data[no_data_pixels] = int(self.data_down.attrs["no_data_mask"])

    def test_shift_disp_row_img(self):
        """
        Test of shift_disp_row_img function
        """
        my_data_down = img_tools.shift_disp_row_img(self.data, 1)
        assert my_data_down.equals(self.data_down)


class TestShiftSubpixImg:
    """Test shift_subpix_img function."""

    @pytest.fixture()
    def monoband_image(self):
        """Create monoband image."""
        data = np.array(
            ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        )

        return xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        ).assign_attrs({"no_data_img": -9999})

    @pytest.fixture()
    def roi_image(self):
        """Create ROI image."""
        data = np.array(
            ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        )

        return xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(2, 7), "col": np.arange(5, 11)}
        ).assign_attrs({"no_data_img": -9999})

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
            pytest.param("roi_image", 2, 1, np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5]), id="monoband image subpix 0.25"),
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
            pytest.param("roi_image", 2, 1, np.array([2.5, 3.5, 4.5, 5.5, 6.5]), id="monoband image subpix 0.25"),
        ],
    )
    def test_row(self, image, subpix, number, expected, request):
        """
        Test shift_subpix_img function for row shift
        """
        shifted_img = img_tools.shift_subpix_img(request.getfixturevalue(image), subpix, True)

        # check if columns coordinates has been shifted
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
