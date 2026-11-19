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
Test P2D_SCORE_THRESHOLD generation.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def img_size():
    return (2, 2)


@pytest.fixture()
def validity_map(criteria_dataarray):
    """3D validity map with every band at zero."""
    return criteria.allocate_validity_dataset(criteria_dataarray)


@pytest.mark.parametrize(
    ["type_measure", "correlation_score", "threshold", "expected_flagged"],
    [
        # type_measure max: skip when correlation score is strictly below the threshold.
        pytest.param("max", np.array([[0.3, 0.8], [0.6, 0.9]]), 0.5, [(0, 0)], id="max below threshold"),
        pytest.param("max", np.array([[0.6, 0.8], [0.7, 0.9]]), 0.5, [], id="max above threshold"),
        # type_measure min: skip when correlation score is strictly above the threshold.
        pytest.param("min", np.array([[3.0, 1.0], [2.0, 4.0]]), 2.0, [(0, 0), (1, 1)], id="min above threshold"),
        pytest.param("min", np.array([[1.0, 1.5], [1.8, 2.0]]), 2.0, [], id="min below threshold"),
    ],
)
def test_apply_score_threshold(validity_map, type_measure, correlation_score, threshold, expected_flagged):
    """Test apply_score_threshold updates band and validity_mask."""
    # In-place update of P2D_SCORE_THRESHOLD and validity_mask bands.
    criteria.apply_score_threshold(
        validity_map["validity"],
        correlation_score,
        threshold,
        type_measure,
    )

    score_band = validity_map["validity"].sel(criteria=Criteria.P2D_SCORE_THRESHOLD.name).data
    for row, col in expected_flagged:
        assert score_band[row, col] == 1
    # Points that meet the threshold keep P2D_SCORE_THRESHOLD at 0.
    for row, col in [(r, c) for r in range(2) for c in range(2) if (r, c) not in expected_flagged]:
        assert score_band[row, col] == 0

    mask_band = validity_map["validity"].sel(criteria="validity_mask").data
    # validity_mask band: 1 where the point was still valid and P2D_SCORE_THRESHOLD was raised.
    for row, col in expected_flagged:
        assert mask_band[row, col] == 1
