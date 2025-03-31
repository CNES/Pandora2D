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
Test create_dataset_from_inputs function.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import numpy as np
import xarray as xr
import pytest

import pandora
from pandora2d import img_tools


class TestReturnedValue:
    """Test expected properties of returned value of create_datasets_from_inputs."""

    @pytest.fixture()
    def result(self, make_input_cfg):
        return img_tools.create_datasets_from_inputs(make_input_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_disparities_are_float32(self, result):
        """Test disparities are float32."""
        assert result.left["col_disparity"].dtype == np.float32
        assert result.left["row_disparity"].dtype == np.float32
        assert result.right["col_disparity"].dtype == np.float32
        assert result.right["row_disparity"].dtype == np.float32

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_use_function_from_pandora(self, mocker, make_input_cfg):
        """Test we use `create_dataset_from_inputs` from pandora.

        We assume this function is well tested in Pandora and that we just need to test that we use it.
        """
        pandora_function = mocker.patch.object(img_tools.pandora_img_tools, "create_dataset_from_inputs")

        img_tools.create_datasets_from_inputs(make_input_cfg)

        pandora_function.assert_has_calls(
            [
                mocker.call(make_input_cfg["left"], None),
                mocker.call(make_input_cfg["right"], None),
            ],
            any_order=True,
        )

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
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

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_disp_band_coordinates(self, result):
        """Test disp_band coordinates is present."""
        np.testing.assert_equal(result.left.coords["band_disp"].data, ["min", "max"])

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
            "left_col_disparity_source",
            "left_row_disparity_source",
            "right_col_disparity_source",
            "right_row_disparity_source",
        ],
        [
            pytest.param(
                {"row_disparity": "correct_grid_for_roi", "col_disparity": "second_correct_grid_for_roi"},
                # second_correct_grid ranges from (height*width) to 0 (so 0 is excluded) ; disp_range is 5 so it is
                # subtracted from min and added to max
                [0.0 + 1 - 5, 375.0 * 450 + 5],
                # correct_grid ranges from 0 to (height*width) (so (height*width) is excluded) ; disp_range is 5
                # so it is subtracted from min and added to max
                [0.0 - 5, 375.0 * 450 - 1 + 5],
                # right disparity is the opposite of left disparity
                [-(375.0 * 450 + 5), -(0.0 + 1 - 5)],
                # right disparity is the opposite of left disparity
                [-(375.0 * 450 - 1 + 5), -(0.0 - 5)],
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                [-2, 2],
                [-2, 4],
                [-2, 2],
                [-4, 2],
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_disparity_source(
        self,
        result,
        left_col_disparity_source,
        left_row_disparity_source,
        right_col_disparity_source,
        right_row_disparity_source,
    ):
        """Test."""

        assert result.left.attrs["col_disparity_source"] == left_col_disparity_source
        assert result.left.attrs["row_disparity_source"] == left_row_disparity_source
        assert result.right.attrs["col_disparity_source"] == right_col_disparity_source
        assert result.right.attrs["row_disparity_source"] == right_row_disparity_source

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
            "expected_left_col_disparity",
            "expected_left_row_disparity",
            "expected_right_col_disparity",
            "expected_right_row_disparity",
        ],
        [
            pytest.param(
                {"row_disparity": "correct_grid_for_roi", "col_disparity": "second_correct_grid_for_roi"},
                # Array of size 2x375x450
                # second_correct_grid ranges from (height*width) to 0 (so 0 is excluded) ; disp_range is 5 so it is
                # subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5),
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5),
                    ]
                ),
                # Array of size 2x375x450
                # correct_grid ranges from 0 to (height*width) (so (height*width) is excluded) ; disp_range is 5
                # so it is subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5),
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5),
                    ]
                ),
                # Array of size 2x375x450
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5),
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5),
                    ]
                ),
                # Array of size 2x375x450
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5),
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5),
                    ]
                ),
                id="Correct disparity grids",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                np.array([np.full((375, 450), -2), np.full((375, 450), 2)]),
                np.array([np.full((375, 450), -2), np.full((375, 450), 4)]),
                np.array([np.full((375, 450), -2), np.full((375, 450), 2)]),
                np.array([np.full((375, 450), -4), np.full((375, 450), 2)]),
                id="Correct disparity dictionaries",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_resulting_disparity_grids(
        self,
        result,
        expected_left_col_disparity,
        expected_left_row_disparity,
        expected_right_col_disparity,
        expected_right_row_disparity,
    ):
        """
        Test the method create_dataset_from_inputs with dictionary and grid disparity

        """

        np.testing.assert_array_equal(result.left["col_disparity"], expected_left_col_disparity)
        np.testing.assert_array_equal(result.left["row_disparity"], expected_left_row_disparity)
        np.testing.assert_array_equal(result.right["col_disparity"], expected_right_col_disparity)
        np.testing.assert_array_equal(result.right["row_disparity"], expected_right_row_disparity)

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
            "expected_left_col_disparity",
            "expected_left_row_disparity",
            "expected_right_col_disparity",
            "expected_right_row_disparity",
            "roi",
        ],
        [
            pytest.param(
                {"row_disparity": "correct_grid_for_roi", "col_disparity": "second_correct_grid_for_roi"},
                # Array of size 2x96x97
                # second_correct_grid ranges from (height*width) to 0 (so 0 is excluded) ; disp_range is 5 so it is
                # subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 8:105],
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 8:105],
                    ]
                ),
                # Array of size 2x96x97
                # correct_grid ranges from 0 to (height*width) (so (height*width) is excluded) ; disp_range is 5
                # so it is subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 8:105],
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 8:105],
                    ]
                ),
                # Array of size 2x96x97
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 8:105],
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 8:105],
                    ]
                ),
                # Array of size 2x96x97
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 8:105],
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 8:105],
                    ]
                ),
                # ROI
                {"col": {"first": 10, "last": 100}, "row": {"first": 10, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity grids with centered ROI",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                np.array([np.full((96, 97), -2), np.full((96, 97), 2)]),
                np.array([np.full((96, 97), -2), np.full((96, 97), 4)]),
                np.array([np.full((96, 97), -2), np.full((96, 97), 2)]),
                np.array([np.full((96, 97), -4), np.full((96, 97), 2)]),
                {"col": {"first": 10, "last": 100}, "row": {"first": 10, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity dictionaries with centered ROI",
            ),
            pytest.param(
                {"row_disparity": "correct_grid_for_roi", "col_disparity": "second_correct_grid_for_roi"},
                # Array of size 2x96x102
                # second_correct_grid ranges from (height*width) to 0 (so 0 is excluded) ; disp_range is 5 so it is
                # subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 348:450],
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 348:450],
                    ]
                ),
                # Array of size 2x96x102
                # correct_grid ranges from 0 to (height*width) (so (height*width) is excluded) ; disp_range is 5
                # so it is subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 348:450],
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 348:450],
                    ]
                ),
                # Array of size 2x96x102
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 348:450],
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 348:450],
                    ]
                ),
                # Array of size 2x96x102
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[7:103, 348:450],
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[7:103, 348:450],
                    ]
                ),
                # ROI
                {"col": {"first": 350, "last": 460}, "row": {"first": 10, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity grids with right overlapping ROI",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                np.array([np.full((96, 102), -2), np.full((96, 102), 2)]),
                np.array([np.full((96, 102), -2), np.full((96, 102), 4)]),
                np.array([np.full((96, 102), -2), np.full((96, 102), 2)]),
                np.array([np.full((96, 102), -4), np.full((96, 102), 2)]),
                {"col": {"first": 350, "last": 460}, "row": {"first": 10, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity dictionaries with right overlapping ROI",
            ),
            pytest.param(
                {"row_disparity": "correct_grid_for_roi", "col_disparity": "second_correct_grid_for_roi"},
                # Array of size 2x103x97
                # second_correct_grid ranges from (height*width) to 0 (so 0 is excluded) ; disp_range is 5 so it is
                # subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[0:103, 8:105],
                        (np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[0:103, 8:105],
                    ]
                ),
                # Array of size 2x103x97
                # correct_grid ranges from 0 to (height*width) (so (height*width) is excluded) ; disp_range is 5
                # so it is subtracted from min and added to max
                np.array(
                    [
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[0:103, 8:105],
                        (np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[0:103, 8:105],
                    ]
                ),
                # Array of size 2x103x97
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) + 5)[0:103, 8:105],
                        -(np.arange(375 * 450, 0, -1, dtype=np.float32).reshape((375, 450)) - 5)[0:103, 8:105],
                    ]
                ),
                # Array of size 2x103x97
                # right is the opposite of left
                np.array(
                    [
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) + 5)[0:103, 8:105],
                        -(np.arange(375 * 450, dtype=np.float32).reshape((375, 450)) - 5)[0:103, 8:105],
                    ]
                ),
                # ROI
                {"col": {"first": 10, "last": 100}, "row": {"first": 0, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity grids with top overlapping ROI",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                np.array([np.full((103, 97), -2), np.full((103, 97), 2)]),
                np.array([np.full((103, 97), -2), np.full((103, 97), 4)]),
                np.array([np.full((103, 97), -2), np.full((103, 97), 2)]),
                np.array([np.full((103, 97), -4), np.full((103, 97), 2)]),
                {"col": {"first": 10, "last": 100}, "row": {"first": 0, "last": 100}, "margins": (2, 3, 4, 2)},
                id="Disparity dictionaries with top overlapping ROI",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_resulting_disparity_grids_with_roi(
        self,
        make_input_cfg,
        expected_left_col_disparity,
        expected_left_row_disparity,
        expected_right_col_disparity,
        expected_right_row_disparity,
        roi,
    ):
        """
        Test the method create_dataset_from_inputs with dictionary and grid disparity with ROI

        """

        make_input_cfg["ROI"] = roi

        datasets = img_tools.create_datasets_from_inputs(make_input_cfg, roi=roi)

        np.testing.assert_array_equal(datasets.left["col_disparity"], expected_left_col_disparity)
        np.testing.assert_array_equal(datasets.left["row_disparity"], expected_left_row_disparity)
        np.testing.assert_array_equal(datasets.right["col_disparity"], expected_right_col_disparity)
        np.testing.assert_array_equal(datasets.right["row_disparity"], expected_right_row_disparity)


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
    def test_fails_when_disparity_is_missing(self, correct_input_cfg, missing, message):
        """
        Description : Test when disparity is not provided.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        for key in missing:
            del correct_input_cfg["input"][key]
        with pytest.raises(KeyError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])
        assert exc_info.value.args[0] == message

    @pytest.mark.parametrize("disparity", [None, 1, 3.14, [-2, 2]])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_disparities_have_wrong_type(self, correct_input_cfg, disparity_key, disparity):
        """
        Description : Test if disparities are not dictionaries or grid in the input section
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        correct_input_cfg["input"][disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])
        assert exc_info.value.args[0] == "The input disparity must be a dictionary."

    @pytest.mark.parametrize("disparity", [{"wrong_init": 2, "range": 2}])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_dict_has_wrong_keys(self, correct_input_cfg, disparity_key, disparity):
        """
        Description : Test dict with wrong keys
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        correct_input_cfg["input"][disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])

        assert exc_info.value.args[0] == "Disparity dictionary should contains keys : init and range"

    @pytest.mark.parametrize("disparity", [{"init": 2.0, "range": 2}])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_init_is_a_float(self, correct_input_cfg, disparity_key, disparity):
        """
        Description : Test if init is a float
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        correct_input_cfg["input"][disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])

        assert exc_info.value.args[0] == "Disparity init should be an integer or a path to a grid"

    @pytest.mark.parametrize("disparity", [{"init": 2, "range": -2}])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_when_range_is_lt_0(self, correct_input_cfg, disparity_key, disparity):
        """
        Description : Test if range is lower than 0
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        correct_input_cfg["input"][disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])

        assert exc_info.value.args[0] == "Disparity range should be an integer greater or equal to 0"

    @pytest.mark.parametrize("disparity", [None, np.nan, np.inf, float("nan"), float("inf")])
    @pytest.mark.parametrize("disparity_key", ["col_disparity", "row_disparity"])
    def test_fails_with_bad_disparity_values(self, correct_input_cfg, disparity_key, disparity):
        """
        Description : Test if the disparity is a dictionary
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        correct_input_cfg["input"][disparity_key] = disparity

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(correct_input_cfg["input"])
        assert exc_info.value.args[0] == "The input disparity must be a dictionary."

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "negative_exploration_grid", "col_disparity": "correct_grid"},
                id="Negative exploration grid for row disparity",
            ),
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "negative_exploration_grid"},
                id="Negative exploration grid for col disparity",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_when_range_band_is_lt_0(self, make_input_cfg):
        """
        Description : Test if range band contains values lower than 0
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """

        with pytest.raises(ValueError) as exc_info:
            img_tools.create_datasets_from_inputs(make_input_cfg)

        assert exc_info.value.args[0] == "Disparity range should be an integer greater or equal to 0"

    def test_create_dataset_from_inputs_with_estimation_step(self, correct_input_cfg):
        """
        test dataset_from_inputs with an estimation step and no disparity range
        """

        configuration_with_estimation = correct_input_cfg
        del configuration_with_estimation["input"]["row_disparity"]
        del configuration_with_estimation["input"]["col_disparity"]
        configuration_with_estimation["pipeline"] = {"estimation": {"estimation_method": "phase_cross_correlation"}}
        result = img_tools.create_datasets_from_inputs(
            correct_input_cfg["input"], estimation_cfg=configuration_with_estimation["pipeline"].get("estimation")
        )

        assert result.left.attrs["col_disparity_source"] == [-9999, -9999]
        assert result.left.attrs["row_disparity_source"] == [-9999, -9999]
        assert result.right.attrs["col_disparity_source"] == [9999, 9999]
        assert result.right.attrs["row_disparity_source"] == [9999, 9999]
