# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test create_dataset_from_inputs function.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import numpy as np
import xarray as xr
import pytest

import pandora
from pandora2d import img_tools


def _make_input_section(left_img_path, right_img_path):
    """This is not a fixture because we want to use it with different scopes."""
    return {
        "left": {
            "img": left_img_path,
            "nodata": -9999,
        },
        "right": {
            "img": right_img_path,
            "nodata": -9999,
        },
        "col_disparity": [-2, 2],
        "row_disparity": [-3, 4],
    }


@pytest.fixture()
def input_section(left_img_path, right_img_path):
    return _make_input_section(left_img_path, right_img_path)


class TestReturnedValue:
    """Test expected properties of returned value of create_datasets_from_inputs."""

    @pytest.fixture()
    def result(self, left_img_path, right_img_path):
        return img_tools.create_datasets_from_inputs(_make_input_section(left_img_path, right_img_path))

    def test_use_function_from_pandora(self, mocker, input_section):
        """Test we use `create_dataset_from_inputs` from pandora.

        We assume this function is well tested in Pandora and that we just need to test that we use it.
        """
        pandora_function = mocker.patch.object(img_tools.pandora_img_tools, "create_dataset_from_inputs")

        img_tools.create_datasets_from_inputs(input_section)

        pandora_function.assert_has_calls(
            [
                mocker.call(input_section["left"], None),
                mocker.call(input_section["right"], None),
            ],
            any_order=True,
        )

    def test_returns_left_and_right_datasets(self, result, left_img_path, right_img_path):
        """Test left and right datasets are returned as namedtuple."""
        assert len(result) == 2
        assert all(isinstance(element, xr.Dataset) for element in result)
        np.testing.assert_equal(
            result.left["im"].data,
            pandora.img_tools.rasterio_open(left_img_path).read(1, out_dtype=np.float32),
        )
        np.testing.assert_equal(
            result.right["im"].data,
            pandora.img_tools.rasterio_open(right_img_path).read(1, out_dtype=np.float32),
        )

    def test_disp_band_coordinates(self, result):
        """Test disp_band coordinates is present."""
        np.testing.assert_equal(result.left.coords["band_disp"].data, ["min", "max"])

    def test_disparity_source(self, result):
        """Test."""
        assert result.left.attrs["col_disparity_source"] == [-2, 2]
        assert result.left.attrs["row_disparity_source"] == [-3, 4]
        assert result.right.attrs["col_disparity_source"] == [-2, 2]
        assert result.right.attrs["row_disparity_source"] == [-4, 3]

    def test_resulting_disparity_grids(self, result):
        """
        Test the method create_dataset_from_inputs with the disparity

        """
        expected_left_col_disparity = np.array([np.full((375, 450), -2), np.full((375, 450), 2)])
        expected_left_row_disparity = np.array([np.full((375, 450), -3), np.full((375, 450), 4)])
        expected_right_col_disparity = np.array([np.full((375, 450), -2), np.full((375, 450), 2)])
        expected_right_row_disparity = np.array([np.full((375, 450), -4), np.full((375, 450), 3)])

        np.testing.assert_array_equal(result.left["col_disparity"], expected_left_col_disparity)
        np.testing.assert_array_equal(result.left["row_disparity"], expected_left_row_disparity)
        np.testing.assert_array_equal(result.right["col_disparity"], expected_right_col_disparity)
        np.testing.assert_array_equal(result.right["row_disparity"], expected_right_row_disparity)


class TestDisparityChecking:
    """Test checks done on disparities."""

    @pytest.mark.parametrize(
        ["missing", "message"],
        [
            pytest.param(["col_disparity"], "`col_disparity` is mandatory.", id="col_disparity"),
            pytest.param(["row_disparity"], "`row_disparity` is mandatory.", id="row_disparity"),
            pytest.param(
                ["col_disparity", "row_disparity"], "`col_disparity` and `row_disparity` are mandatory.", id="both"
            ),
        ],
    )
    def test_fails_when_disparity_is_missing(self, input_section, missing, message):
        """Test when disparity is not provided."""
        for key in missing:
            del input_section[key]
        with pytest.raises(KeyError) as exc_info:
            img_tools.create_datasets_from_inputs(input_section)
        assert exc_info.value.args[0] == message

    @pytest.mark.parametrize("disparity", [None, 1, 3.14, "grid_path"])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_disparities_are_not_lists_or_tuples(self, input_section, disparity_key, disparity):
        """Test."""
        input_section[disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(input_section)
        assert exc_info.value.args[0] == "Disparity should be iterable of length 2"

    @pytest.mark.parametrize("disparity", [None, np.nan, np.inf, float("nan"), float("inf")])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_with_bad_disparity_values(self, input_section, disparity_key, disparity):
        """Test."""
        input_section[disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(input_section)
        assert exc_info.value.args[0] == "Disparity should be iterable of length 2"

    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_disparity_max_lt_disparity_min(self, input_section, disparity_key):
        """Test."""
        input_section[disparity_key] = [8, -10]
        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(input_section)
        assert exc_info.value.args[0] == "Min disparity (8) should be lower than Max disparity (-10)"

    def test_create_dataset_from_inputs_with_estimation_step(self, input_section):
        """
        test dataset_from_inputs with an estimation step and no disparity range
        """

        configuration_with_estimation = {"input": input_section}
        del configuration_with_estimation["input"]["row_disparity"]
        del configuration_with_estimation["input"]["col_disparity"]
        configuration_with_estimation["pipeline"] = {"estimation": {"estimation_method": "phase_cross_correlation"}}
        result = img_tools.create_datasets_from_inputs(
            input_section, estimation_cfg=configuration_with_estimation["pipeline"].get("estimation")
        )

        assert result.left.attrs["col_disparity_source"] == [-9999, -9999]
        assert result.left.attrs["row_disparity_source"] == [-9999, -9999]
        assert result.right.attrs["col_disparity_source"] == [9999, 9999]
        assert result.right.attrs["row_disparity_source"] == [9999, 9999]
