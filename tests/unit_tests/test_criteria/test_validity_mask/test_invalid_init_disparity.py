#  Copyright (c) 2026. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Test various P2D_INVALID_INIT_DISPARITY band.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def criteria_name():
    return Criteria.P2D_INVALID_INIT_DISPARITY.name


@pytest.mark.parametrize("img_size", [(6, 8)])
@pytest.mark.parametrize(
    ["slice_criteria", "step", "subpix", "expected"],
    [
        pytest.param(
            np.s_[0:0, :, :, :],  # Empty slice
            [1, 1],
            1,
            np.zeros((6, 8)),
            id="Empty slice",
        ),
        pytest.param(
            np.s_[:, :, :, :],  # Full slice
            [1, 1],
            1,
            np.ones((6, 8)),
            id="Full slice",
        ),
        pytest.param(
            np.s_[0:2, 2:6, :, :],  # This criteria is added for all disparities
            [1, 1],
            1,
            np.array(
                [
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            id="Classic case",
        ),
        pytest.param(
            np.s_[0:2, 2:6, :, :],  # This criteria is added for all disparities
            [1, 1],
            4,
            np.array(
                [
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            id="Subpix=4",
        ),
        pytest.param(
            np.s_[0:2, 0:2, :, :],  # This criteria is added for all disparities
            [2, 3],
            1,
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            id="Step=[2,3]",
        ),
    ],
)
def test_invalid_init_disparity(
    criteria_dataarray, criteria_name, slice_criteria, expected, row_disparity_source, col_disparity_source
):
    """
    Test that the produced invalid init disparity band is correct.
    """

    criteria_dataarray[slice_criteria] = Criteria.P2D_INVALID_INIT_DISPARITY
    result = (
        criteria.get_validity_dataset(criteria_dataarray, row_disparity_source, col_disparity_source)
        .sel(criteria=criteria_name)["validity"]
        .data
    )

    np.testing.assert_array_equal(result, expected)
