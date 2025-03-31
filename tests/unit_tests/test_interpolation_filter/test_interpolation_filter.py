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
"""
Test the interpolation filter module.
"""

import numpy as np
import pytest

from pandora2d.interpolation_filter import AbstractFilter


@pytest.mark.parametrize(
    ["resampling_area", "row_coeff", "col_coeff", "expected_value"],
    [
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            1.5,
            id="Shift of 0.5 in columns and in rows with identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            1.5,
            id="Shift of 0.5 in columns with identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            1.0,
            id="Shift of 0.5 in rows with identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([-0.0703125, 0.8671875, 0.2265625, -0.0234375]),
            1.25,
            id="Shift of 0.25 in columns with identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([-0.0703125, 0.8671875, 0.2265625, -0.0234375]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            1.0,
            id="Shift of 0.25 in rows with identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            1.5,
            id="Shift of 0.5 in columns and in rows with identical columns in resampling area",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            1.0,
            id="Shift of 0.5 in columns with identical columns in resampling area",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            1.5,
            id="Shift of 0.5 in rows with identical columns in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 4, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            2.625,
            id="Shift of 0.5 in columns with 3/4 identical rows in resampling area",
        ),
        pytest.param(
            np.array([[0, 1, 2, 3], [0, 1, 4, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            np.array([-0.0625, 0.5625, 0.5625, -0.0625]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            1.0,
            id="Shift of 0.5 in rows with 3/4 identical rows in resampling area",
        ),
    ],
)
def test_apply(resampling_area, row_coeff, col_coeff, expected_value):
    """
    Test the apply method
    """
    assert AbstractFilter.apply(resampling_area, row_coeff, col_coeff) == expected_value


@pytest.fixture
def centered_one_image():
    return np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


@pytest.fixture
def identical_rows_image():
    return np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])


@pytest.fixture
def identical_cols_image():
    return np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])


@pytest.fixture
def almost_identical_rows_image():
    return np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 10, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])


@pytest.fixture
def almost_identical_cols_image():
    return np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 10, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])


@pytest.mark.parametrize("filter_method", ["bicubic"])
@pytest.mark.parametrize(
    ["image", "positions_col", "positions_row", "expected_values"],
    [
        pytest.param(
            "centered_one_image",
            np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5]),
            [0.31640625, 0.5625, 0.31640625, 0.5625, 1.0, 0.5625, 0.31640625, 0.5625, 0.31640625],
            id="Interpolation around the center and precision=0.5",
        ),
        pytest.param(
            "centered_one_image",
            np.array([1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25]),
            np.array([1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25]),
            [
                0.75201416015625,
                0.8671875,
                0.75201416015625,
                0.8671875,
                1.0,
                0.8671875,
                0.75201416015625,
                0.8671875,
                0.75201416015625,
            ],
            id="Interpolation around the center and precision=0.25",
        ),
        pytest.param(
            "centered_one_image",
            np.array([1.99999999, 1.99999999, 1.99999999, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25]),
            np.array([1.99999999, 2.0, 2.25, 1.9999999, 2.0, 2.25, 1.99999999, 2.0, 2.25]),
            [
                0.9999809489561501,
                0.9999904744327068,
                0.867179239547113,
                0.9999904744327068,
                1.0,
                0.8671875,
                0.867179239547113,
                0.8671875,
                0.75201416015625,
            ],
            id="Best candidate at the center and subpixel shift close to 1",
        ),
        pytest.param(
            "identical_rows_image",
            np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5]),
            [1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5],
            id="Identical rows and precision=0.5",
        ),
        pytest.param(
            "identical_rows_image",
            np.array([1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25]),
            np.array([1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25]),
            [1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25],
            id="Identical rows and precision=0.25",
        ),
        pytest.param(
            "identical_cols_image",
            np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5]),
            [1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5],
            id="Identical columns and precision=0.5",
        ),
        pytest.param(
            "identical_cols_image",
            np.array([1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25]),
            np.array([1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25]),
            [1.75, 2.0, 2.25, 1.75, 2.0, 2.25, 1.75, 2.0, 2.25],
            id="Identical columns and precision=0.25",
        ),
        pytest.param(
            "almost_identical_rows_image",
            np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5]),
            [4.03125, 6.0, 4.03125, 6.5, 10.0, 6.5, 5.03125, 7.0, 5.03125],
            id="4/5 identical rows and precision=0.5",
        ),
        pytest.param(
            "almost_identical_cols_image",
            np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5]),
            [4.03125, 6.5, 5.03125, 6.0, 10.0, 7.0, 4.03125, 6.5, 5.03125],
            id="4/5 identical columns and precision=0.5",
        ),
    ],
)
def test_interpolate(filter_method, image, positions_col, positions_row, expected_values, request):
    """
    Test the interpolate method
    """

    assert (
        AbstractFilter({"method": filter_method}).interpolate(  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated
            request.getfixturevalue(image), (positions_col, positions_row)
        )
        == expected_values
    )
