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
Test get_min_max_disp_from_dicts function.
"""

from collections.abc import Iterable
from pathlib import Path

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name,unused-argument,too-many-arguments,too-many-positional-arguments
import numpy as np
import pandora
import pytest
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


class TestGetMinMaxDispFromDicts:
    """
    Test the get_min_max_disp_from_dicts method
    """

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
                (375, 450),
                (375, 450),
                {"row": 0, "col": 0},
                [1, 1],
                -9999,
                id="Step=[1,1] and invalid_disp = -9999",
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
                np.inf,
                {"col": {"first": 11, "last": 15}, "row": {"first": 11, "last": 15}, "margins": (2, 1, 1, 2)},
                np.array(
                    [
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, -3.0, -3.0, -3.0, -3.0, -3.0, np.inf],
                        [np.inf, np.inf, -5.0, -5.0, -5.0, -5.0, -5.0, np.inf],
                        [np.inf, np.inf, -2.0, -2.0, -2.0, -2.0, -2.0, np.inf],
                        [np.inf, np.inf, -3.0, -3.0, -3.0, -3.0, -3.0, np.inf],
                        [np.inf, np.inf, -5.0, -5.0, -5.0, -5.0, -5.0, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                    ]
                ),
                np.array(
                    [
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, 7.0, 7.0, 7.0, 7.0, 7.0, np.inf],
                        [np.inf, np.inf, 5.0, 5.0, 5.0, 5.0, 5.0, np.inf],
                        [np.inf, np.inf, 8.0, 8.0, 8.0, 8.0, 8.0, np.inf],
                        [np.inf, np.inf, 7.0, 7.0, 7.0, 7.0, 7.0, np.inf],
                        [np.inf, np.inf, 5.0, 5.0, 5.0, 5.0, 5.0, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
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
            pytest.param(
                (8, 9),
                (8, 9),
                {"row": 14, "col": 31},
                [1, 1],
                np.nan,
                {"col": {"first": 31, "last": 39}, "row": {"first": 16, "last": 17}, "margins": (1, 1, 1, 1)},
                np.array(
                    [
                        [np.nan, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, np.nan],
                        [np.nan, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, np.nan],
                        [np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, np.nan],
                        [np.nan, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, np.nan],
                    ],
                ),
                np.array(
                    [
                        [np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, np.nan],
                        [np.nan, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, np.nan],
                        [np.nan, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan],
                        [np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, np.nan],
                    ],
                ),
                id="Step=[1,1] - Equivalent of directory disparity with segment mode (offset is (14,31))",
            ),
            pytest.param(
                (4, 3),
                (4, 3),
                {"row": 14, "col": 31},
                [2, 3],
                np.nan,
                {"col": {"first": 31, "last": 39}, "row": {"first": 16, "last": 17}, "margins": (1, 1, 1, 1)},
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, -5.0, np.nan, np.nan, -5.0, np.nan, np.nan, -5.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, -2.0, np.nan, np.nan, -2.0, np.nan, np.nan, -2.0, np.nan, np.nan, np.nan],
                    ],
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, 5.0, np.nan, np.nan, 5.0, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, 8.0, np.nan, np.nan, 8.0, np.nan, np.nan, 8.0, np.nan, np.nan, np.nan],
                    ],
                ),
                id="Step=[2,3] - Equivalent of directory disparity with segment mode (offset is (14,31))",
            ),
            pytest.param(
                (4, 3),
                (4, 3),
                {"row": 16, "col": 31},
                [3, 2],
                np.nan,
                {"col": {"first": 31, "last": 36}, "row": {"first": 16, "last": 17}, "margins": (1, 1, 1, 1)},
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, -3, np.nan, -3, np.nan, -3, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ],
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, 7, np.nan, 7, np.nan, 7, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ],
                ),
                id="Step=[2,3] - Equivalent of directory disparity with segment mode (offset is (16,31))",
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
                    "row_disparity": "subpix_grid",
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
                        [np.nan, np.nan, -4.0, -4.0, -4.0, -4.0, -4.0, np.nan],
                        [np.nan, np.nan, -2.0, -2.0, -2.0, -2.0, -2.0, np.nan],
                        [np.nan, np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, np.nan],
                        [np.nan, np.nan, -4.0, -4.0, -4.0, -4.0, -4.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan],
                        [np.nan, np.nan, 6.0, 6.0, 6.0, 6.0, 6.0, np.nan],
                        [np.nan, np.nan, 8.0, 8.0, 8.0, 8.0, 8.0, np.nan],
                        [np.nan, np.nan, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan],
                        [np.nan, np.nan, 6.0, 6.0, 6.0, 6.0, 6.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                id="Subpixel initial disparity grid",
            ),
        ],
    )
    def test_str_subpix_disparity_with_roi(
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
        subpix_grid_data,
    ):
        """
        Test the get_min_max_disp_from_dicts method with string initial disparity containing subpixel values and ROI
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
        correct_grid_min = np.round(subpix_grid_data) - make_input_cfg["row_disparity"]["range"]
        correct_grid_max = np.round(subpix_grid_data) + make_input_cfg["row_disparity"]["range"]

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


class TestComputeValidDisparityGridIndex:  # pylint: disable=too-few-public-methods
    """
    Test the compute_valid_disparity_grid_index method
    """

    @pytest.mark.parametrize(
        [
            "dataset_coordinates",
            "disparity_grid_shape",
            "origin",
            "step",
            "ground_truth_dataset",
            "ground_truth_disparity_grid",
        ],
        [
            pytest.param(
                np.arange(10),
                10,
                0,
                1,
                np.arange(10),
                np.arange(10),
                id="Classic case",
            ),
            pytest.param(
                np.arange(10),
                5,
                0,
                1,
                np.arange(5),
                np.arange(5),
                id="Smaller disparity grid",
            ),
            pytest.param(
                np.arange(10),  # (rows, cols)
                5,
                3,
                1,
                np.arange(3, 8),
                np.arange(5),
                id="Smaller disparity grid with offset",
            ),
            pytest.param(
                np.arange(20),
                5,
                3,
                2,
                np.arange(3, 13, 2),
                np.arange(5),
                id="Smaller disparity grid with offset and step",
            ),
            pytest.param(
                np.arange(2, 7),
                10,
                0,
                1,
                np.arange(5),
                np.arange(2, 7),
                id="Smaller dataset",
            ),
            pytest.param(
                np.arange(2, 7),
                10,
                4,
                1,
                np.arange(2, 5),
                np.arange(
                    3,
                ),
                id="Smaller dataset with offset",
            ),
            pytest.param(
                np.arange(2, 7),
                10,
                4,
                3,
                np.arange(2, 3),
                np.arange(1),
                id="Smaller dataset with offset and step",
            ),
            pytest.param(
                np.arange(3, 8),
                10,
                9,
                1,
                np.arange(0),
                np.arange(0),
                id="No intersection between dataset and disparity grid",
            ),
        ],
    )
    def test_compute_valid_disparity_grid_index(
        self,
        dataset_coordinates,
        disparity_grid_shape,
        origin,
        step,
        ground_truth_dataset,
        ground_truth_disparity_grid,
    ):
        """
        Test the compute_valid_disparity_grid_index method
        """

        disp_min_max_index, disp_data_index = img_tools.compute_valid_disparity_grid_index(
            dataset_coordinates,
            disparity_grid_shape,
            origin,
            step,
        )

        np.testing.assert_equal(disp_min_max_index, ground_truth_dataset)
        np.testing.assert_equal(disp_data_index, ground_truth_disparity_grid)
