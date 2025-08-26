# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
Test allocate_cost_volumes method from Matching cost
"""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
import xarray as xr
from rasterio import Affine

from pandora2d import matching_cost
from pandora2d.img_tools import create_datasets_from_inputs


def test_allocate_cost_volume(left_stereo_object, right_stereo_object):
    """
    Test the allocate cost_volumes function
    """

    # generated data for the test
    np_data = np.empty((3, 3, 3, 5))
    np_data.fill(np.nan)

    c_row = [0, 1, 2]
    c_col = [0, 1, 2]

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(0, 4 + 1)
    disparity_range_row = np.arange(-2, 0 + 1)

    cost_volumes_test = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_row", "disp_col"], np_data)},
        coords={"row": row, "col": col, "disp_row": disparity_range_row, "disp_col": disparity_range_col},
    )

    cost_volumes_test.attrs["measure"] = "zncc_python"
    cost_volumes_test.attrs["window_size"] = 3
    cost_volumes_test.attrs["type_measure"] = "max"
    cost_volumes_test.attrs["subpixel"] = 1
    cost_volumes_test.attrs["offset_row_col"] = 1
    cost_volumes_test.attrs["crs"] = None
    cost_volumes_test.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    cost_volumes_test.attrs["col_disparity_source"] = [-1, 3]
    cost_volumes_test.attrs["row_disparity_source"] = [-2, 0]
    cost_volumes_test.attrs["no_data_img"] = -9999
    cost_volumes_test.attrs["no_data_mask"] = 1
    cost_volumes_test.attrs["valid_pixels"] = 0
    cost_volumes_test.attrs["step"] = [1, 1]
    cost_volumes_test.attrs["disparity_margins"] = None

    # data by function compute_cost_volume
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "zncc_python", "window_size": 3}}}
    matching_cost_matcher = matching_cost.PandoraMatchingCostMethods(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate(img_left=left_stereo_object, img_right=right_stereo_object, cfg=cfg)
    cost_volumes_fun = matching_cost_matcher.compute_cost_volumes(
        img_left=left_stereo_object, img_right=right_stereo_object
    )

    # After deleting the calls to the pandora cv_masked and validity_mask methods in matching cost step,
    # only points that are not no data in the ground truth are temporarily checked
    # because some invalid points are no longer equal to nan in the calculated cost volumes.
    valid_mask = ~np.isnan(cost_volumes_test["cost_volumes"].data)

    # check that the generated xarray dataset is equal to the ground truth
    np.testing.assert_array_equal(
        cost_volumes_fun["cost_volumes"].data[valid_mask], cost_volumes_test["cost_volumes"].data[valid_mask]
    )
    assert cost_volumes_fun.attrs == cost_volumes_test.attrs


@pytest.fixture()
def monoband_image():
    """Create monoband image."""
    data = np.array(
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
    )

    return xr.Dataset(
        {
            "im": (["row", "col"], data),
            "row_disparity": (["band_disp", "row", "col"], np.ones((2, *data.shape)) * [[[1]], [[3]]]),
            "col_disparity": (["band_disp", "row", "col"], np.ones((2, *data.shape)) * [[[1]], [[3]]]),
        },
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
            "band_disp": ["min", "max"],
        },
    ).assign_attrs({"no_data_img": -9999})


@pytest.fixture()
def roi_image(monoband_image):
    """Create ROI image."""
    return monoband_image.assign_coords(row=np.arange(2, 7), col=np.arange(5, 11))


class TestShiftedRightImagesAffectation:
    """Test shift_subpix_img function."""

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc_python"])
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
    def test_only_row(self, image, configuration, matching_cost_object, number, expected, request):
        """
        Test row shift.
        """
        img_left = img_right = request.getfixturevalue(image)
        matching_cost_matcher = matching_cost_object(configuration["pipeline"]["matching_cost"])

        matching_cost_matcher.allocate(img_left=img_left, img_right=img_right, cfg=configuration)

        # check if row coordinates has been shifted
        np.testing.assert_array_equal(expected, matching_cost_matcher.shifted_right_images[number].row)

    @pytest.mark.parametrize("matching_cost_method", ["mutual_information"])
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
    def test_both_rows_and_columns(
        self,
        image,
        matching_cost_object,
        configuration,
        subpix,
        number,
        expected_row,
        expected_col,
        request,
    ):
        """
        Test both row and col shifts.
        """

        img_left = img_right = request.getfixturevalue(image)
        matching_cost_matcher = matching_cost_object(configuration["pipeline"]["matching_cost"])

        matching_cost_matcher.allocate(img_left=img_left, img_right=img_right, cfg=configuration)

        assert len(matching_cost_matcher.shifted_right_images) == subpix**2
        np.testing.assert_array_equal(expected_row, matching_cost_matcher.shifted_right_images[number].row)
        np.testing.assert_array_equal(expected_col, matching_cost_matcher.shifted_right_images[number].col)


class TestCvFloatPrecision:
    """
    Test that the cost volumes is allocated with the right type
    """

    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc_python", "zncc"])
    @pytest.mark.parametrize("float_precision", ["float32", "f", "f4"])
    def test_cost_volumes_float_precision(self, input_config, matching_cost_config, matching_cost_object):
        """
        Test that the cost volumes is allocated in np.float32
        """

        cfg = {"input": input_config, "pipeline": {"matching_cost": matching_cost_config}}

        img_left, img_right = create_datasets_from_inputs(input_config)

        matching_cost_test = matching_cost_object(matching_cost_config)

        matching_cost_test.allocate(img_left=img_left, img_right=img_right, cfg=cfg)

        assert matching_cost_test.cost_volumes["cost_volumes"].dtype == np.float32

    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    @pytest.mark.parametrize("float_precision", ["float64", "d", "f8"])
    def test_cost_volumes_double_precision(self, input_config, matching_cost_config, matching_cost_object):
        """
        Test that the cost volumes is allocated in np.float64
        """

        cfg = {"input": input_config, "pipeline": {"matching_cost": matching_cost_config}}

        img_left, img_right = create_datasets_from_inputs(input_config)

        matching_cost_test = matching_cost_object(matching_cost_config)

        matching_cost_test.allocate(img_left=img_left, img_right=img_right, cfg=cfg)

        assert matching_cost_test.cost_volumes["cost_volumes"].dtype == np.float64
