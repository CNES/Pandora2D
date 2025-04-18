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

"""Fixtures linked to criteria tests"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import xarray as xr

from pandora2d import matching_cost
from pandora2d.constants import Criteria
from pandora2d.img_tools import add_disparity_grid


@pytest.fixture()
def img_size():
    row = 10
    col = 13
    return (row, col)


@pytest.fixture()
def disparity_cfg():
    """Return (disp_row, disp_col)"""
    return {"init": 1, "range": 2}, {"init": -1, "range": 4}


@pytest.fixture()
def subpix():
    return 1


@pytest.fixture()
def step():
    return [1, 1]


@pytest.fixture()
def start_point():
    return [0, 0]


@pytest.fixture()
def valid_pixels():
    return 0


@pytest.fixture()
def no_data_mask():
    return 1


@pytest.fixture()
def window_size():
    return 1


@pytest.fixture()
def matching_cost_cfg(window_size, subpix, step):
    return {"matching_cost_method": "ssd", "window_size": window_size, "subpix": subpix, "step": step}


@pytest.fixture()
def image(img_size, disparity_cfg, valid_pixels, no_data_mask, start_point):
    """Make image"""
    row, col = img_size
    row_disparity, col_disparity = disparity_cfg
    data = np.random.uniform(0, row * col, (row, col))

    return xr.Dataset(
        {
            "im": (["row", "col"], data),
            "msk": (["row", "col"], np.zeros_like(data)),
        },
        coords={"row": np.arange(start_point[0], data.shape[0]), "col": np.arange(start_point[1], data.shape[1])},
        attrs={
            "no_data_img": -9999,
            "valid_pixels": valid_pixels,
            "no_data_mask": no_data_mask,
            "crs": None,
            "invalid_disparity": np.nan,
        },
    ).pipe(add_disparity_grid, col_disparity, row_disparity)


@pytest.fixture()
def cost_volumes(matching_cost_cfg, image):
    """Compute a cost_volumes"""
    matching_cost_ = matching_cost.PandoraMatchingCostMethods(matching_cost_cfg)

    matching_cost_.allocate(img_left=image, img_right=image, cfg=matching_cost_cfg)
    return matching_cost_.compute_cost_volumes(img_left=image, img_right=image)


@pytest.fixture()
def criteria_dataarray(img_size, subpix, step, start_point):
    """
    Create a criteria dataarray
    """
    row = np.arange(start_point[0], img_size[0], step[0])
    col = np.arange(start_point[1], img_size[1], step[1])
    shape = (len(row), len(col), len(np.arange(-1, 3.25, 1 / subpix)), len(np.arange(-5, 3.25, 1 / subpix)))
    return xr.DataArray(
        np.full(shape, Criteria.VALID),
        coords={
            "row": row,
            "col": col,
            "disp_row": np.arange(-1, 3.25, 1 / subpix),
            "disp_col": np.arange(-5, 3.25, 1 / subpix),
        },
        dims=["row", "col", "disp_row", "disp_col"],
    )
