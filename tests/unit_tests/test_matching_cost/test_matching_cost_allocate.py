# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
    np_data = np.empty((3, 3, 3, 5))
    np_data.fill(np.nan)

    c_row = [0, 1, 2]
    c_col = [0, 1, 2]

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(0, 4 + 1)
    disparity_range_row = np.arange(-2, 0 + 1)

    cost_volumes_test = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_row", "disp_col"], np_data)},
        coords={"row": row, "col": col, "disp_row": disparity_range_row, "disp_col": disparity_range_col},
    )

    cost_volumes_test.attrs["measure"] = "zncc"
    cost_volumes_test.attrs["window_size"] = 3
    cost_volumes_test.attrs["type_measure"] = "max"
    cost_volumes_test.attrs["subpixel"] = 1
    cost_volumes_test.attrs["offset_row_col"] = 1
    cost_volumes_test.attrs["crs"] = None
    cost_volumes_test.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    cost_volumes_test.attrs["col_disparity_source"] = [-1, 3]
    cost_volumes_test.attrs["row_disparity_source"] = [-2, 0]
    cost_volumes_test.attrs["no_data_img"] = -9999
    cost_volumes_test.attrs["no_data_mask"] = 1
    cost_volumes_test.attrs["valid_pixels"] = 0
    cost_volumes_test.attrs["step"] = [1, 1]
    cost_volumes_test.attrs["disparity_margins"] = None

    # data by function compute_cost_volume
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3}}}
    matching_cost_matcher = matching_cost.PandoraMatchingCostMethods(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate(img_left=left_stereo_object, img_right=right_stereo_object, cfg=cfg)
    cost_volumes_fun = matching_cost_matcher.compute_cost_volumes(
        img_left=left_stereo_object, img_right=right_stereo_object
    )

    # After deleting the calls to the pandora cv_masked and validity_mask methods in matching cost step,
    # only points that are not no data in the ground truth are temporarily checked
    # because some invalid points are no longer equal to nan in the calculated cost volumes.
    valid_mask = ~np.isnan(cost_volumes_test["cost_volumes"].data)

    # check that the generated xarray dataset is equal to the ground truth
    np.testing.assert_array_equal(
        cost_volumes_fun["cost_volumes"].data[valid_mask], cost_volumes_test["cost_volumes"].data[valid_mask]
    )
    assert cost_volumes_fun.attrs == cost_volumes_test.attrs
