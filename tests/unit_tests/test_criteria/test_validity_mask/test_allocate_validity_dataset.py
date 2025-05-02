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
#

"""
Test that the validity xr.Dataset is correctly allocated.
"""

import pytest
import numpy as np

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.mark.parametrize(
    ["make_cost_volumes"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            id="Classic case",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 2,
                "window_size": 1,
            },
            id="Subpix=2",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 4,
                "window_size": 1,
            },
            id="Subpix=4",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_allocate_validity_dataset(make_cost_volumes):
    """
    Test the allocate_validity_dataset method
    """

    cost_volumes = make_cost_volumes

    criteria_dataarray = cost_volumes.criteria

    allocated_validity_mask = criteria.allocate_validity_dataset(criteria_dataarray)

    assert allocated_validity_mask.sizes["row"] == criteria_dataarray.sizes["row"]
    assert allocated_validity_mask.sizes["col"] == criteria_dataarray.sizes["col"]
    # The dimension 'criteria' is the same size as the Enum Criteria -1
    # because there is a band for each criteria except the 'Valid' and the 'P2D_DISPARITY_UNPROCESSED'
    # and a band for the global 'validity_mask'.
    assert allocated_validity_mask.sizes["criteria"] == len(Criteria.__members__) - 1
