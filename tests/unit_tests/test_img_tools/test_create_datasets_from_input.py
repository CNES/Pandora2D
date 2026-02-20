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

from collections.abc import Iterable
from pathlib import Path

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name,unused-argument,too-many-arguments,too-many-positional-arguments, too-many-lines
import numpy as np
import pandora
import pytest
import xarray as xr
from numpy.typing import DTypeLike

from pandora2d import img_tools
from pandora2d.margins import Margins
from pandora2d.types import Origin, Step


def build_data(
    shape: tuple[int, ...],
    default_value: int | float,
    assignments: Iterable[tuple[tuple[slice, ...], int | float]],
    dtype: DTypeLike = np.float32,
):
    """
    Build numpy array with default value and assign slices with others.

    :param shape: shape of the returned array
    :param default_value: fill array with ``default_value`` where assignments are not defined.
    :param assignments: couples of slice value where to fill array with.
    :param dtype: dtype of the returned array.
    :return: numpy array build with ``default_value`` and ``assignments``

    example:
    >>> build_data((6, 5), np.nan, ((np.s_[0, 0], 10.898), (np.s_[1:5:2, 1:4], 7.0), (np.s_[2:5:2, 1:4], 5.0)))
    array([[10.898,    nan,    nan,    nan,    nan],
           [   nan,  7.   ,  7.   ,  7.   ,    nan],
           [   nan,  5.   ,  5.   ,  5.   ,    nan],
           [   nan,  7.   ,  7.   ,  7.   ,    nan],
           [   nan,  5.   ,  5.   ,  5.   ,    nan],
           [   nan,    nan,    nan,    nan,    nan]], dtype=float32)
    """
    result = np.full(shape, default_value, dtype)
    for indices, value in assignments:
        result[indices] = value
    return result


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

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "step",
        ],
        [
            [(10, 10), (10, 10), {"row": 11, "col": 11}, [2, 2]],
            [(13, 15), (13, 15), {"row": 14, "col": 15}, [2, 2]],
        ],
    )
    def test_disparities_from_roi(
        self,
        attributes,
        correct_grid_shape,
        second_correct_grid_shape,
        origin_coordinates,
        step,
        disparity_range,
        left_img_shape,
        correct_grid_data,
        second_correct_grid_data,
        make_input_cfg,
    ):
        """
        Test disparities generated from a previous run with ROI (smaller than image with a step)

        We expect that attributes with step exists and was integrated into the input section.
        """
        # We need to transform directory path to grid path:
        make_input_cfg["row_disparity"]["init"] = str(Path(make_input_cfg["row_disparity"]["init"]) / "row_map.tif")
        make_input_cfg["col_disparity"]["init"] = str(Path(make_input_cfg["col_disparity"]["init"]) / "col_map.tif")

        # Margins value is arbitrary
        roi = {
            "col": {
                "first": origin_coordinates["col"],
                "last": origin_coordinates["col"] + correct_grid_shape[1] * step[1],
            },
            "row": {
                "first": origin_coordinates["row"],
                "last": origin_coordinates["row"] + correct_grid_shape[0] * step[0],
            },
            "margins": (1, 1, 1, 1),
        }

        result = img_tools.create_datasets_from_inputs(make_input_cfg, roi=roi, attributes=attributes)

        row_offset = origin_coordinates["row"] - result.left.row.data[0]
        col_offset = origin_coordinates["col"] - result.left.col.data[0]

        assert result.left.sizes["row"], result.left.sizes["col"] == left_img_shape
        assert result.right.sizes["row"], result.right.sizes["col"] == left_img_shape
        np.testing.assert_equal(
            result.left["row_disparity"]
            .sel(band_disp="min")
            .data[
                row_offset : row_offset + correct_grid_shape[0] * step[0] : step[0],
                col_offset : col_offset + correct_grid_shape[1] * step[1] : step[1],
            ]
            + disparity_range,
            correct_grid_data,
        )
        np.testing.assert_equal(
            result.left["row_disparity"]
            .sel(band_disp="max")
            .data[
                row_offset : row_offset + correct_grid_shape[0] * step[0] : step[0],
                col_offset : col_offset + correct_grid_shape[1] * step[1] : step[1],
            ]
            - disparity_range,
            correct_grid_data,
        )
        np.testing.assert_equal(
            result.left["col_disparity"]
            .sel(band_disp="min")
            .data[
                row_offset : row_offset + second_correct_grid_shape[0] * step[0] : step[0],
                col_offset : col_offset + second_correct_grid_shape[1] * step[1] : step[1],
            ]
            + disparity_range,
            second_correct_grid_data,
        )
        np.testing.assert_equal(
            result.left["col_disparity"]
            .sel(band_disp="max")
            .data[
                row_offset : row_offset + second_correct_grid_shape[0] * step[0] : step[0],
                col_offset : col_offset + second_correct_grid_shape[1] * step[1] : step[1],
            ]
            - disparity_range,
            second_correct_grid_data,
        )

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize("no_data_disp", [69])
    def test_no_data_disp(self, result, no_data_disp):
        """no_data_disp should be present in attributes."""
        assert result.left["col_disparity"].attrs["no_data"] == no_data_disp
        assert result.left["row_disparity"].attrs["no_data"] == no_data_disp

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Correct disparity grids",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize("no_data_disp", [np.nan])
    def test_no_data_disp_is_nan(self, result, no_data_disp):
        """no_data_disp should be present in attributes."""
        assert np.isnan(result.left["col_disparity"].attrs["no_data"])
        assert np.isnan(result.left["row_disparity"].attrs["no_data"])


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


class TestGetMinMaxDispFromDicts:
    """
    Test the get_min_max_disp_from_dicts method
    """

    # Other invalid disparity values will be tested
    # after the tickets for processing invalid disp have been done
    @pytest.fixture()
    def invalid_disp(self):
        """
        Invalid disparity value
        """
        return np.nan

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        ["correct_grid_shape", "second_correct_grid_shape", "origin_coordinates", "step", "invalid_init_disp"],
        [
            pytest.param(
                (375, 450),
                (375, 450),
                {"row": 0, "col": 0},
                [1, 1],
                np.nan,
                id="Step=[1,1]",
            ),
            pytest.param(
                (63, 113),
                (63, 113),
                {"row": 0, "col": 0},
                [6, 4],
                np.nan,
                id="Step=[6,4]",
            ),
        ],
    )
    def test_str_disparity_without_roi(
        self,
        make_input_cfg,
        correct_grid_shape,
        second_correct_grid_shape,
        origin_coordinates,
        step,
        invalid_init_disp,
        correct_grid_data,
    ):
        """
        Test the get_min_max_disp_from_dicts method with string initial disparity
        """

        # We need to transform directory path to grid path:
        make_input_cfg["row_disparity"]["init"] = str(Path(make_input_cfg["row_disparity"]["init"]) / "row_map.tif")

        dataset = pandora.img_tools.create_dataset_from_inputs(make_input_cfg["left"])

        # We test for row_disparity, the behavior for col_disparity is the same.
        disp_min_max, disp_interval, nodata = img_tools.get_min_max_disp_from_dicts(
            dataset,
            make_input_cfg["row_disparity"],
            Origin(origin_coordinates["row"], origin_coordinates["col"]),
            Step(step[0], step[1]),
            invalid_init_disp,
        )

        # Preparation of data for a more readable comparison in the assertion
        correct_grid_min = correct_grid_data - make_input_cfg["row_disparity"]["range"]
        correct_grid_max = correct_grid_data + make_input_cfg["row_disparity"]["range"]

        np.testing.assert_equal(disp_min_max[0, :: step[0], :: step[1]], correct_grid_min)
        np.testing.assert_equal(disp_min_max[1, :: step[0], :: step[1]], correct_grid_max)
        assert disp_interval[0] == np.nanmin(correct_grid_min)
        assert disp_interval[1] == np.nanmax(correct_grid_max)
        assert nodata is None

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "correct_grid",
                    "col_disparity": "second_correct_grid",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "step",
            "invalid_init_disp",
            "roi",
            "gt_disparity_min",
            "gt_disparity_max",
        ],
        [
            pytest.param(
                (5, 5),
                (5, 5),
                {"row": 11, "col": 11},
                [1, 1],
                np.nan,
                {"col": {"first": 11, "last": 15}, "row": {"first": 11, "last": 15}, "margins": (2, 1, 1, 2)},
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, np.nan],
                        [np.nan, np.nan, -5.0, -5.0, -5.0, -5.0, -5.0, np.nan],
                        [np.nan, np.nan, -2.0, -2.0, -2.0, -2.0, -2.0, np.nan],
                        [np.nan, np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, np.nan],
                        [np.nan, np.nan, -5.0, -5.0, -5.0, -5.0, -5.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan],
                        [np.nan, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, np.nan],
                        [np.nan, np.nan, 8.0, 8.0, 8.0, 8.0, 8.0, np.nan],
                        [np.nan, np.nan, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan],
                        [np.nan, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                id="Step=[1,1] - Equivalent of directory disparity",
            ),
            pytest.param(
                (2, 3),
                (2, 3),
                {"row": 16, "col": 31},
                [3, 2],
                np.nan,
                {"col": {"first": 31, "last": 37}, "row": {"first": 16, "last": 22}, "margins": (1, 2, 1, 2)},
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, -3.0, np.nan, -3.0, np.nan, -3.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, -5.0, np.nan, -5.0, np.nan, -5.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, 7.0, np.nan, 7.0, np.nan, 7.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, 5.0, np.nan, 5.0, np.nan, 5.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                id="Step=[3,2] - Equivalent of directory disparity",
            ),
            pytest.param(
                (375, 450),
                (375, 450),
                {"row": 0, "col": 0},
                [1, 1],
                np.nan,
                {"col": {"first": 11, "last": 15}, "row": {"first": 11, "last": 15}, "margins": (2, 1, 1, 2)},
                np.array(
                    [
                        [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                        [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                        [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                        [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                        [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    ]
                ),
                np.array(
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                        [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                        [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                    ]
                ),
                id="Step=[1,1] - Equivalent of tif file disparity",
            ),
        ],
    )
    def test_str_disparity_with_roi(
        self,
        make_input_cfg,
        correct_grid_shape,
        second_correct_grid_shape,
        origin_coordinates,
        step,
        invalid_init_disp,
        roi,
        gt_disparity_min,
        gt_disparity_max,
        correct_grid_data,
    ):
        """
        Test the get_min_max_disp_from_dicts method with string initial disparity and roi
        """

        make_input_cfg["ROI"] = roi

        dataset = pandora.img_tools.create_dataset_from_inputs(make_input_cfg["left"], roi=roi)

        # We test for row_disparity, the behavior for col_disparity is the same.
        disp_min_max, disp_interval, nodata = img_tools.get_min_max_disp_from_dicts(
            dataset,
            make_input_cfg["row_disparity"],
            Origin(origin_coordinates["row"], origin_coordinates["col"]),
            Step(step[0], step[1]),
            invalid_init_disp,
        )

        # Preparation of data for a more readable comparison in the assertion
        correct_grid_min = correct_grid_data - make_input_cfg["row_disparity"]["range"]
        correct_grid_max = correct_grid_data + make_input_cfg["row_disparity"]["range"]

        np.testing.assert_equal(disp_min_max[0, :, :], gt_disparity_min)
        np.testing.assert_equal(disp_min_max[1, :, :], gt_disparity_max)
        assert disp_interval[0] == np.nanmin(correct_grid_min)
        assert disp_interval[1] == np.nanmax(correct_grid_max)
        assert nodata is None

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "constant_initial_disparity",
                    "col_disparity": "second_constant_initial_disparity",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        ["roi"],
        [
            pytest.param(
                None,
                id="Without ROI",
            ),
            pytest.param(
                {"col": {"first": 11, "last": 20}, "row": {"first": 11, "last": 20}, "margins": (2, 3, 4, 2)},
                id="With ROI",
            ),
        ],
    )
    def test_int_initial_disparity(
        self,
        make_input_cfg,
        roi,
    ):
        """
        Test the get_min_max_disp_from_dicts method with int initial disparity
        """

        make_input_cfg["ROI"] = roi

        dataset = pandora.img_tools.create_dataset_from_inputs(make_input_cfg["left"], roi=roi)

        # We test for row_disparity, the behavior for col_disparity is the same.
        disp_min_max, disp_interval, no_data = img_tools.get_min_max_disp_from_dicts(
            dataset,
            make_input_cfg["row_disparity"],
            Origin(0, 0),
            Step(1, 1),
            np.nan,  # When a directory is not used as input disparity, the invalid initial disparity value is np.nan.
        )

        correct_grid_min = (
            np.full(dataset["im"].data.shape, make_input_cfg["row_disparity"]["init"])
            - make_input_cfg["row_disparity"]["range"]
        )
        correct_grid_max = (
            np.full(dataset["im"].data.shape, make_input_cfg["row_disparity"]["init"])
            + make_input_cfg["row_disparity"]["range"]
        )

        np.testing.assert_equal(disp_min_max[0, :, :], correct_grid_min)
        np.testing.assert_equal(disp_min_max[1, :, :], correct_grid_max)
        assert disp_interval[0] == np.nanmin(correct_grid_min)
        assert disp_interval[1] == np.nanmax(correct_grid_max)
        assert no_data is None


class TestGetMinMaxDispFromDictsNoData:
    """
    Test the gestion of nodata in get_min_max_disp_from_dicts method
    """

    @pytest.fixture
    def second_correct_grid_data(self, second_correct_grid_shape, second_min, second_max, no_data_disp):
        """Override second_correct_grid_data to use new values"""
        return build_data(
            second_correct_grid_shape,
            np.mean([second_max, second_min], dtype=int),
            [
                (np.s_[:, 0::4], second_min),
                (np.s_[:, 1::4], second_max),
                (np.s_[:, 2::4], -np.inf),
                (np.s_[:, 3::4], no_data_disp),
            ],
        )

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "correct_grid",
                    "col_disparity": "second_correct_grid",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "step",
            "invalid_init_disp",
            "roi",
        ],
        [
            pytest.param(
                (5, 5),
                (5, 5),
                {"row": 11, "col": 11},
                [1, 1],
                np.nan,
                {"col": {"first": 11, "last": 15}, "row": {"first": 11, "last": 15}, "margins": (2, 1, 1, 2)},
                id="Step=[1,1] - Equivalent of directory disparity",
            ),
        ],
    )
    @pytest.mark.parametrize(
        [
            "second_min",
            "second_max",
            "no_data_disp",
        ],
        [
            pytest.param(
                -34,
                999,
                -999,
            ),
            pytest.param(
                -34,
                11,
                700,
            ),
        ],
    )
    def test_no_data_value_does_not_influence_disp_interval_computation(
        self,
        make_input_cfg,
        correct_grid_shape,
        second_correct_grid_shape,
        origin_coordinates,
        step,
        invalid_init_disp,
        roi,
        correct_grid_data,
        second_min,
        second_max,
        no_data_disp,
        second_correct_grid_data,
    ):
        """
        When the extreme values of a disparity grid are flagged as nodata in the grid metadata they must not be
        used to compute disp_interval, and they must remain untouched in the grid.
        """

        make_input_cfg["ROI"] = roi
        margins = Margins(*roi["margins"])

        dataset = pandora.img_tools.create_dataset_from_inputs(make_input_cfg["left"], roi=roi)

        # We test for col_disparity, the behavior for row_disparity is the same.
        disp_min_max, disp_interval, nodata = img_tools.get_min_max_disp_from_dicts(
            dataset,
            make_input_cfg["col_disparity"],
            Origin(origin_coordinates["row"], origin_coordinates["col"]),
            Step(step[0], step[1]),
            invalid_init_disp,
        )
        disparity_range = make_input_cfg["col_disparity"]["range"]
        assert disp_interval == [second_min - disparity_range, second_max + disparity_range]
        # According to second_correct_grid_data, check that no_disp_data values are still in place.
        assert (
            disp_min_max[:, margins.up : -margins.down, margins.left + 3 : -margins.right : 4] == no_data_disp
        ).all()
        assert nodata == no_data_disp

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "correct_grid",
                    "col_disparity": "second_correct_grid",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "step",
            "invalid_init_disp",
            "roi",
        ],
        [
            pytest.param(
                (5, 5),
                (5, 5),
                {"row": 11, "col": 11},
                [1, 1],
                np.nan,
                {"col": {"first": 11, "last": 15}, "row": {"first": 11, "last": 15}, "margins": (2, 1, 1, 2)},
                id="Step=[1,1] - Equivalent of directory disparity",
            ),
        ],
    )
    @pytest.mark.parametrize(
        [
            "second_min",
            "second_max",
            "no_data_disp",
        ],
        [
            pytest.param(
                -34,
                999,
                -999,
            ),
            pytest.param(
                -34,
                11,
                700,
            ),
        ],
    )
    def test_no_data_value_does_not_influence_disp_interval_computation_with_right_image(
        self,
        make_input_cfg,
        correct_grid_shape,
        second_correct_grid_shape,
        origin_coordinates,
        step,
        invalid_init_disp,
        roi,
        correct_grid_data,
        second_min,
        second_max,
        no_data_disp,
        second_correct_grid_data,
    ):
        """
        When the extreme values of a disparity grid are flagged as nodata in the grid metadata they must not be
        used to compute disp_interval and they must remain untouched in the grid.

        When right argument of get_min_max_disp_from_dicts is set to True, oposite of min and max are used.
        """

        make_input_cfg["ROI"] = roi
        margins = Margins(*roi["margins"])

        dataset = pandora.img_tools.create_dataset_from_inputs(make_input_cfg["left"], roi=roi)

        # We test for col_disparity, the behavior for row_disparity is the same.
        disp_min_max, disp_interval, nodata = img_tools.get_min_max_disp_from_dicts(
            dataset,
            make_input_cfg["col_disparity"],
            Origin(origin_coordinates["row"], origin_coordinates["col"]),
            Step(step[0], step[1]),
            invalid_init_disp,
            right=True,
        )
        disparity_range = make_input_cfg["col_disparity"]["range"]
        assert disp_interval == [-second_max - disparity_range, -second_min + disparity_range]
        # According to second_correct_grid_data, check that no_disp_data values are still in place.
        assert (
            disp_min_max[:, margins.up : -margins.down, margins.left + 3 : -margins.right : 4] == no_data_disp
        ).all()
        assert nodata == no_data_disp


@pytest.mark.parametrize(
    ["disp_data", "nodata", "expected"],
    [
        pytest.param(np.array([1, 2]), None, [True, True], id="Nothing to filter"),
        pytest.param(np.array([1, np.inf]), None, [True, False], id="inf"),
        pytest.param(np.array([-np.inf, 2]), None, [False, True], id="-inf"),
        pytest.param(np.array([np.nan, 2]), None, [False, True], id="nan"),
        pytest.param(np.array([3, 2]), 3, [False, True], id="value"),
        pytest.param(np.array([3, np.inf, np.nan, 2]), 3, [False, False, False, True], id="mix"),
    ],
)
def test_build_usable_data_mask(disp_data, nodata, expected):
    """Unusable values are masked to False."""
    result = img_tools.build_usable_data_mask(disp_data, nodata)
    assert (result == expected).all()


@pytest.mark.parametrize(
    ["init_value", "range_value", "expected"],
    [
        pytest.param(1, 3, (-2, 4), id="int"),
        pytest.param(1.0, 3, (-2, 4), id="float"),
        pytest.param(np.array([3, 0]), 7, (-7, 10), id="no nan"),
        pytest.param(np.array([3, np.nan, 0]), 7, (-7, 10), id="nan"),
    ],
)
def test_get_extrema_disparity(init_value, range_value, expected):
    """NaNs are filtered."""
    result = img_tools.get_extrema_disparity(init_value, range_value)
    assert result == expected


class TestNodataFiltering:
    """Nodata in disparity grids should not influence margin determination."""

    @pytest.fixture
    def second_correct_grid_data(self, second_correct_grid_shape):
        """second_correct_grid_data override to include inf and nan."""
        data = np.full(second_correct_grid_shape, 5.0)
        data[:, 1::4] = -21
        data[:, 2::4] = -np.inf
        data[:, 3::4] = np.nan
        return data

    @pytest.mark.parametrize(
        ["second_correct_grid_shape", "nodata", "expected"], [((1, 5), None, [[[5, -21, np.nan, np.nan, 5]]])]
    )
    def test_get_initial_disparity_str(
        self, create_disparity_grid_fixture, second_correct_grid_data, second_correct_grid_shape, nodata, expected
    ):
        """Test invalid values are replaced by NaNs"""
        disparity = create_disparity_grid_fixture(second_correct_grid_data, 2, "disparity.tiff", nodata=nodata)

        result = img_tools.get_initial_disparity(disparity)

        np.testing.assert_array_equal(result, expected)

    def test_get_initial_disparity_int(self):
        """Test init is returned"""
        disparity = {"init": 1, "range": 2}

        result = img_tools.get_initial_disparity(disparity)

        assert result == 1
