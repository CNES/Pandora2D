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
Test mask_invalid_init_disparity function.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


class TestMaskInvalidInitDisparity:
    """
    Test mask_invalid_init_disparity function.
    """

    @pytest.fixture()
    def image_test(self, request):
        """
        Image used to test mask_invalid_init_disparity function.
        """
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize("img_size", [(4, 5)])
    @pytest.mark.parametrize(
        ["image_test", "no_data_disp", "expected"],
        [
            # pylint: disable=line-too-long
            pytest.param(
                "image_variable_disp",  # image fixture with variable disparity grids
                1,
                np.array(
                    [
                        # fmt: off
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                        # fmt: on
                    ]
                ),
                id="No data disp = 1",
            ),
            pytest.param(
                "image_variable_disp",  # image fixture with variable disparity grids
                3,
                np.array(
                    [
                        # fmt: off
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, 0, 0, 0],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, 0, 0, 0],
                        # fmt: on
                    ]
                ),
                id="No data disp = 3",
            ),
            pytest.param(
                "image_nan_disp",  # image fixture with nan in disparity grids
                np.nan,
                np.array(
                    [
                        # fmt: off
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, 0, 0, 0],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, 0, 0, 0],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                        # fmt: on
                    ]
                ),
                id="No data disp is np.nan",
            ),
            pytest.param(
                "image_inf_disp",  # image fixture with inf in disparity grids
                np.inf,
                np.array(
                    [
                        # fmt: off
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                            [0, 0, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY, Criteria.P2D_INVALID_INIT_DISPARITY],
                        # fmt: on
                    ]
                ),
                id="No data disp is np.inf",
            ),
        ],
        indirect=["image_test"],
    )
    def test_mask_invalid_init_disparity(self, image_test, criteria_dataarray, no_data_disp, expected):
        """
        Test mask_invalid_init_disparity function.
        """

        image_test["row_disparity"].attrs["no_data"] = no_data_disp
        image_test["col_disparity"].attrs["no_data"] = no_data_disp

        # image_test fixtures (image_variable_disp, image_nan_disp and image_inf_disp)
        # are located in the file tests/unit_tests/test_criteria/conftest.py.
        # Comments in the fixtures show the minimum and maximum disparities for each point.
        criteria.mask_invalid_init_disparity(criteria_dataarray, image_test["row_disparity"])
        criteria.mask_invalid_init_disparity(criteria_dataarray, image_test["col_disparity"])

        # P2D_INVALID_INIT_DISPARITY is raised independently of disparity values
        for i in range(criteria_dataarray.data.shape[2]):
            for j in range(criteria_dataarray.data.shape[3]):
                assert np.all(criteria_dataarray.data[:, :, i, j] == expected)
