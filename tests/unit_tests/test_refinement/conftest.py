# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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

"""
Conftest for dichotomy cpp and python versions module.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import xarray as xr

from pandora2d import refinement
from pandora2d.matching_cost import PandoraMatchingCostMethods


@pytest.fixture()
def matching_cost_method():
    return "sad"


@pytest.fixture()
def row_step():
    return 1


@pytest.fixture()
def col_step():
    return 1


@pytest.fixture()
def step(row_step, col_step):
    return [row_step, col_step]


@pytest.fixture()
def window_size():
    return 1


@pytest.fixture()
def min_row():
    return 0


@pytest.fixture()
def max_row():
    return 1


@pytest.fixture()
def image_row_coordinates(min_row, max_row):
    return np.arange(min_row, max_row + 1)


@pytest.fixture()
def row_coordinates_with_step(min_row, max_row, step):
    """Row coordinates used into disparity map and cost volume"""
    return np.arange(min_row, max_row + 1, step[0])


@pytest.fixture()
def min_col():
    return 0


@pytest.fixture()
def max_col():
    return 2


@pytest.fixture()
def image_col_coordinates(min_col, max_col):
    return np.arange(min_col, max_col + 1)


@pytest.fixture()
def col_coordinates_with_step(min_col, max_col, step):
    """Col coordinates used into disparity map and cost volume"""
    return np.arange(min_col, max_col + 1, step[1])


@pytest.fixture()
def min_disparity_row():
    return 2


@pytest.fixture()
def max_disparity_row():
    return 7


@pytest.fixture()
def min_disparity_col():
    return -2


@pytest.fixture()
def max_disparity_col():
    return 3


@pytest.fixture()
def type_measure():
    return "max"


# Once the criteria for identifying extremas at the edge of disparity ranges has been implemented,
# this fixture could possibly be removed.
@pytest.fixture()
def left_img(
    image_row_coordinates,
    image_col_coordinates,
    min_disparity_row,
    max_disparity_row,
    min_disparity_col,
    max_disparity_col,
):
    """
    Creates a left image dataset
    """

    img = xr.Dataset(
        {"im": (["row", "col"], np.full((image_row_coordinates.size, image_col_coordinates.size), 0))},
        coords={"row": image_row_coordinates, "col": image_col_coordinates},
    )

    d_min_col = np.full((image_row_coordinates.size, image_col_coordinates.size), min_disparity_col)
    d_max_col = np.full((image_row_coordinates.size, image_col_coordinates.size), max_disparity_col)
    d_min_row = np.full((image_row_coordinates.size, image_col_coordinates.size), min_disparity_row)
    d_max_row = np.full((image_row_coordinates.size, image_col_coordinates.size), max_disparity_row)

    # Once the variable disparity grids have been introduced into pandora2d,
    # it will be possible to call a method such as add_disparity_grid
    # to complete img with uniform or non-uniform disparity grids.

    # Here, it is completed by hand because the disparity range is even.
    img["col_disparity"] = xr.DataArray(
        np.array([d_min_col, d_max_col]),
        dims=["band_disp", "row", "col"],
        coords={"band_disp": ["min", "max"]},
    )

    img["row_disparity"] = xr.DataArray(
        np.array([d_min_row, d_max_row]),
        dims=["band_disp", "row", "col"],
        coords={"band_disp": ["min", "max"]},
    )

    img.attrs.update(
        {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "row_disparity_source": [np.min(d_min_row), np.max(d_max_row)],
            "col_disparity_source": [np.min(d_min_col), np.max(d_max_col)],
        }
    )

    return img


@pytest.fixture()
def right_img(left_img):
    data = np.roll(left_img["im"].data, 2)
    data[:2] = 0
    return left_img.copy(
        deep=True,
        data={"im": data, "row_disparity": left_img["row_disparity"], "col_disparity": left_img["col_disparity"]},
    )


@pytest.fixture()
def zeros_cost_volumes(matching_cost_config, left_img, right_img):
    matching_cost = PandoraMatchingCostMethods(matching_cost_config)
    matching_cost.allocate(left_img, right_img, matching_cost_config)
    return matching_cost.cost_volumes


@pytest.fixture()
def cost_volumes(zeros_cost_volumes):
    """Pandora2d cost volumes fake data."""
    zeros_cost_volumes["cost_volumes"].data[:] = np.arange(zeros_cost_volumes["cost_volumes"].data.size).reshape(
        zeros_cost_volumes["cost_volumes"].data.shape
    )
    return zeros_cost_volumes


@pytest.fixture()
def invalid_disparity():
    return np.nan


@pytest.fixture()
def disp_map(invalid_disparity, row_coordinates_with_step, col_coordinates_with_step):
    """Fake disparity maps with alternating values."""
    row = np.full(row_coordinates_with_step.size * col_coordinates_with_step.size, 4.0)
    row[::2] = 5
    col = np.full(row_coordinates_with_step.size * col_coordinates_with_step.size, 0.0)
    col[::2] = 1
    return xr.Dataset(
        {
            "row_map": (["row", "col"], row.reshape((row_coordinates_with_step.size, col_coordinates_with_step.size))),
            "col_map": (["row", "col"], col.reshape((row_coordinates_with_step.size, col_coordinates_with_step.size))),
        },
        coords={
            "row": row_coordinates_with_step,
            "col": col_coordinates_with_step,
        },
        attrs={"invalid_disp": invalid_disparity},
    )


@pytest.fixture()
def iterations():
    return 2


@pytest.fixture()
def filter_name():
    return "bicubic"


@pytest.fixture()
def config_dichotomy(iterations, filter_name):
    return {
        "iterations": iterations,
        "filter": {"method": filter_name},
    }


@pytest.fixture()
def dichotomy_python_instance(config_dichotomy):
    config_dichotomy["refinement_method"] = "dichotomy_python"
    return refinement.dichotomy.DichotomyPython(config_dichotomy)


@pytest.fixture()
def dichotomy_cpp_instance(config_dichotomy):
    config_dichotomy["refinement_method"] = "dichotomy"
    return refinement.dichotomy_cpp.Dichotomy(config_dichotomy)
