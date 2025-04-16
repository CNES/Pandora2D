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
Test various P2D_PEAK_ON_EDGE generation.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def row_map(img_size, disparity_cfg):
    """
    Row disparity map used for tests
    """

    row_map = np.full(img_size, 2)

    # row_map[0,0] is equal to the minimum of the row disparity range
    row_map[0, 0] = disparity_cfg[0]["init"] - disparity_cfg[0]["range"]
    # row_map[3,3] is equal to the maximum of the row disparity range
    row_map[3, 3] = disparity_cfg[0]["init"] + disparity_cfg[0]["range"]
    return row_map


@pytest.fixture()
def col_map(img_size, disparity_cfg):
    """
    Col disparity map used for tests
    """

    col_map = np.full(img_size, -1)

    # col_map[0,0] is equal to the maximum of the col disparity range
    col_map[0, 0] = disparity_cfg[1]["init"] + disparity_cfg[1]["range"]
    # col_map[4,5] is equal to the minimum of the col disparity range
    col_map[4, 5] = disparity_cfg[1]["init"] - disparity_cfg[1]["range"]
    return col_map


@pytest.fixture()
def row_map_full_peak(img_size, disparity_cfg):
    """
    Row disparity map with only peak on edges used for tests
    """

    # row_map is filled with the minimum of the row disparity range
    row_map = np.full(img_size, disparity_cfg[0]["init"] - disparity_cfg[0]["range"])
    return row_map


@pytest.fixture()
def col_map_full_peak(img_size, disparity_cfg):
    """
    Col disparity map with only peak on edges used for tests
    """

    # col_map is filled with the maximum of the col disparity range
    col_map = np.full(img_size, disparity_cfg[1]["init"] + disparity_cfg[1]["range"])
    return col_map


@pytest.fixture()
def map_without_peak(img_size):
    """
    Disparity map without peak on edges
    """

    return np.full(img_size, 1)


@pytest.fixture()
def validity_map(criteria_dataarray):
    """
    3D validity map
    """

    # row_map[3,3] is equal to the maximum of the row disparity range
    # and validity_map[3,3] is valid
    validity_map = criteria.allocate_validity_dataset(criteria_dataarray)

    # row_map[0,0] is equal to the minimum of the row disparity range
    # and validity_map[0,0] is already invalid
    validity_map["validity"].sel(criteria="validity_mask").data[0, 0] = 2

    # col_map[4,5] is equal to the minimum of the col disparity range
    # and validity_map[4,5] is partially valid
    validity_map["validity"].sel(criteria="validity_mask").data[4, 5] = 1

    return validity_map


def test_apply_peak_on_edge(validity_map, image, cost_volumes, row_map, col_map):
    """
    Test the apply_peak_on_edge method
    """

    cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

    # Apply P2D_PEAK_ON_EDGE on validity_map
    criteria.apply_peak_on_edge(validity_map["validity"], image, cost_volumes_coords, row_map, col_map)

    # P2D_PEAK_ON_EDGE band is equal to 1 on [0,0], [4,5] and [3,3] points
    assert validity_map["validity"][0, 0].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name).data == 1
    assert validity_map["validity"][4, 5].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name).data == 1
    assert validity_map["validity"][3, 3].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name).data == 1

    # Global validity_mask is equal to 2 on [0,0] point that was already equal to 2
    # before apply_peak_on_edge
    assert validity_map["validity"][0, 0].sel(criteria="validity_mask").data == 2
    # Global validity_mask is equal to 1 on [4,5] point that was already equal to 1
    # before apply_peak_on_edge
    assert validity_map["validity"][4, 5].sel(criteria="validity_mask").data == 1
    # Global validity_mask is equal to 1 on [3,3] point that was equal to 0
    # before apply_peak_on_edge
    assert validity_map["validity"][3, 3].sel(criteria="validity_mask").data == 1


@pytest.mark.parametrize(
    ["drow_map", "dcol_map"],
    [
        pytest.param("row_map_full_peak", "col_map_full_peak", id="Row and col disparity maps full of peaks"),
        pytest.param("row_map_full_peak", "col_map", id="Row map full of peaks"),
        pytest.param("map_without_peak", "col_map_full_peak", id="Col map full of peaks"),
    ],
)
def test_apply_peak_on_edge_full_peak_map(validity_map, image, cost_volumes, drow_map, dcol_map, request):
    """
    Test the apply_peak_on_edge method with disparity maps full of peaks on edges
    """

    cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

    # Apply P2D_PEAK_ON_EDGE on validity_map
    criteria.apply_peak_on_edge(
        validity_map["validity"],
        image,
        cost_volumes_coords,
        request.getfixturevalue(drow_map),
        request.getfixturevalue(dcol_map),
    )

    # P2D_PEAK_ON_EDGE band is equal to 1 on every points
    assert (validity_map["validity"].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name).data == 1).all()

    # validity_map is now full of 1, except for the point [0,0] which is still equal to 2
    assert (validity_map["validity"].sel(criteria="validity_mask").data >= 1).all()
    assert validity_map["validity"][0, 0].sel(criteria="validity_mask").data == 2
    # [0,0] is the unique point equal to 2 in the validity_mask band
    assert (validity_map["validity"].sel(criteria="validity_mask").data == 2).sum() == 1


def test_apply_peak_on_edge_without_peak(validity_map, image, cost_volumes, map_without_peak):
    """
    Test the apply_peak_on_edge method with maps without peaks on edges
    """

    cost_volumes_coords = (cost_volumes.row.values, cost_volumes.col.values)

    # Apply P2D_PEAK_ON_EDGE on full_valid_map
    criteria.apply_peak_on_edge(
        validity_map["validity"], image, cost_volumes_coords, map_without_peak, map_without_peak
    )

    # P2D_PEAK_ON_EDGE band is equal to 0 on every points
    assert (validity_map["validity"].sel(criteria=Criteria.P2D_PEAK_ON_EDGE.name).data == 0).all()

    # No P2D_PEAK_ON_EDGE so validity map stay inchanged
    assert validity_map["validity"].sel(criteria="validity_mask").data[0, 0] == 2
    # [0,0] is the unique point equal to 2 in the validity_mask band
    assert (validity_map["validity"].sel(criteria="validity_mask").data == 2).sum() == 1

    assert validity_map["validity"].sel(criteria="validity_mask").data[4, 5] == 1
    # [4,5] is the unique point equal to 1 in the validity_mask band
    assert (validity_map["validity"].sel(criteria="validity_mask").data == 1).sum() == 1
