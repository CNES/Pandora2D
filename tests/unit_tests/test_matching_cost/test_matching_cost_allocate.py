# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test allocate_cost_volumes method from Matching cost
"""

# pylint: disable=redefined-outer-name
import numpy as np
import xarray as xr
from rasterio import Affine

from pandora2d import matching_cost


def test_allocate_cost_volume(left_stereo_object, right_stereo_object):
    """
    Test the allocate cost_volumes function
    """

    # generated data for the test
    np_data = np.empty((3, 3, 2, 2))
    np_data.fill(np.nan)

    c_row = [0, 1, 2]
    c_col = [0, 1, 2]

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(-1, 0 + 1)
    disparity_range_row = np.arange(-1, 0 + 1)

    # Create the cost volume
    if np_data is None:
        np_data = np.zeros((len(row), len(col), len(disparity_range_col), len(disparity_range_row)), dtype=np.float32)

    cost_volumes_test = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], np_data)},
        coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
    )

    cost_volumes_test.attrs["measure"] = "zncc"
    cost_volumes_test.attrs["window_size"] = 3
    cost_volumes_test.attrs["type_measure"] = "max"
    cost_volumes_test.attrs["subpixel"] = 1
    cost_volumes_test.attrs["offset_row_col"] = 1
    cost_volumes_test.attrs["cmax"] = 1
    cost_volumes_test.attrs["crs"] = None
    cost_volumes_test.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    cost_volumes_test.attrs["band_correl"] = None
    cost_volumes_test.attrs["col_disparity_source"] = [0, 1]
    cost_volumes_test.attrs["row_disparity_source"] = [-1, 0]
    cost_volumes_test.attrs["no_data_img"] = -9999
    cost_volumes_test.attrs["no_data_mask"] = 1
    cost_volumes_test.attrs["valid_pixels"] = 0
    cost_volumes_test.attrs["step"] = [1, 1]
    cost_volumes_test.attrs["disparity_margins"] = None

    # data by function compute_cost_volume
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3}}}
    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        cfg=cfg,
    )
    cost_volumes_fun = matching_cost_matcher.compute_cost_volumes(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )

    # check that the generated xarray dataset is equal to the ground truth
    np.testing.assert_array_equal(cost_volumes_fun["cost_volumes"].data, cost_volumes_test["cost_volumes"].data)
    assert cost_volumes_fun.attrs == cost_volumes_test.attrs
