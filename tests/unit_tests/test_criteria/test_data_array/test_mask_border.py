#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
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

"""
Test mask_border method.
"""

import numpy as np
import pytest
import xarray as xr

from pandora2d import criteria
from pandora2d.constants import Criteria


class TestMaskBorder:
    """Test mask_border method."""

    def test_null_offset(self, image, criteria_dataarray):
        """offset = 0, no raise P2D_LEFT_BORDER criteria"""
        make_criteria_copy = criteria_dataarray.copy(deep=True)
        criteria.mask_border(image, 0, criteria_dataarray)

        # Check criteria_dataarray has not changed
        xr.testing.assert_equal(criteria_dataarray, make_criteria_copy)
        # Check the P2D_LEFT_BORDER criteria does not raise
        assert np.all(criteria_dataarray.data[:, :, :, :] != Criteria.P2D_LEFT_BORDER)

    @pytest.mark.parametrize("img_size", [(5, 6)])
    @pytest.mark.parametrize(
        ["offset", "step", "expected"],
        [
            pytest.param(
                1,
                [1, 1],
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
                id="offset=1 and no step",
            ),
            pytest.param(
                2,
                [1, 1],
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
                id="offset=2 and no step",
            ),
            pytest.param(
                3,
                [1, 1],
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
                id="offset=3 and no step",
            ),
            pytest.param(
                1,
                [1, 2],
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 1, 1],
                    ]
                ),
                id="offset=1 and step=[1,2]",
            ),
            pytest.param(
                1,
                [3, 1],
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 1],
                    ]
                ),
                id="offset=1 and step=[3,1]",
            ),
            pytest.param(
                2,
                [2, 3],
                np.array(
                    [
                        [1, 1],
                        [1, 0],
                        [1, 1],
                    ]
                ),
                id="offset=2 and step=[2,3]",
            ),
        ],
    )
    def test_variable_offset(self, image, criteria_dataarray, offset, expected):
        """
        With mask_border, the P2D_LEFT_BORDER criteria is raised on the border.

        Example :
        offset = 1

        For this image :          1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8
                                  1 2 3 4 5 6 7 8

        and a criteria_dataarray :  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0
                                  0 0 0 0 0 0 0 0

        the result is :           1 1 1 1 1 1 1 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 0 0 0 0 0 0 1
                                  1 1 1 1 1 1 1 1
        """
        criteria.mask_border(image, offset, criteria_dataarray)

        # P2D_LEFT_BORDER is raised independently of disparity values
        for i in range(criteria_dataarray.data.shape[2]):
            for j in range(criteria_dataarray.data.shape[3]):
                assert np.all(criteria_dataarray.data[:, :, i, j] == expected)
