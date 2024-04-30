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
"""
Test the interpolation filter module.
"""

import numpy as np
import pytest

from pandora2d import interpolation_filter


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
    assert interpolation_filter.AbstractFilter.apply(resampling_area, row_coeff, col_coeff) == expected_value
