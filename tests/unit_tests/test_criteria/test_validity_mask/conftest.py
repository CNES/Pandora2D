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
Fixtures.
"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import xarray as xr

from pandora2d import matching_cost
from pandora2d.img_tools import add_disparity_grid


@pytest.fixture()
def make_image():
    """
    Create image dataset
    """

    def inner(row_disparity, col_disparity, mask_data, roi=None):
        """Make image"""
        row, col = (6, 8)

        data = np.random.uniform(0, row * col, (row, col))

        row_coords = np.arange(roi["row"]["first"], roi["row"]["last"] + 1) if roi else np.arange(data.shape[0])
        col_coords = np.arange(roi["col"]["first"], roi["col"]["last"] + 1) if roi else np.arange(data.shape[1])

        return xr.Dataset(
            {
                "im": (["row", "col"], data),
                "msk": (["row", "col"], mask_data),
            },
            coords={"row": row_coords, "col": col_coords},
            attrs={
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "invalid_disparity": np.nan,
            },
        ).pipe(add_disparity_grid, col_disparity, row_disparity)

    return inner


@pytest.fixture()
def make_cost_volumes(make_image, request):
    """
    Instantiate a matching_cost and compute cost_volumes
    """

    cfg = {
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": request.param["window_size"],
                "step": request.param["step"],
                "subpix": request.param["subpix"],
            }
        }
    }

    disp_row = request.param["row_disparity"]
    disp_col = request.param["col_disparity"]

    img_left = make_image(disp_row, disp_col, request.param["msk_left"], request.param.get("roi"))
    img_right = make_image(disp_row, disp_col, request.param["msk_right"], request.param.get("roi"))

    matching_cost_object = matching_cost.MatchingCostRegistry.get(
        cfg["pipeline"]["matching_cost"]["matching_cost_method"]
    )
    matching_cost_ = matching_cost_object(cfg["pipeline"]["matching_cost"])

    matching_cost_.allocate(img_left=img_left, img_right=img_right, cfg=cfg)

    cost_volumes = matching_cost_.compute_cost_volumes(img_left=img_left, img_right=img_right)

    return cost_volumes
