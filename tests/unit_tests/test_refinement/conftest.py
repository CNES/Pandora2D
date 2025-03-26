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

from pandora2d.matching_cost import PandoraMatchingCostMethods
from pandora2d import refinement


@pytest.fixture()
def rows():
    return np.arange(2)


@pytest.fixture()
def cols():
    return np.arange(3)


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


@pytest.fixture()
def subpixel():
    return 1


# Once the criteria for identifying extremas at the edge of disparity ranges has been implemented,
# this fixture could possibly be removed.
@pytest.fixture()
def left_img(rows, cols, min_disparity_row, max_disparity_row, min_disparity_col, max_disparity_col):
    """
    Creates a left image dataset
    """

    img = xr.Dataset(
        {"im": (["row", "col"], np.full((rows.size, cols.size), 0))},
        coords={"row": rows, "col": cols},
    )

    d_min_col = np.full((rows.size, cols.size), min_disparity_col)
    d_max_col = np.full((rows.size, cols.size), max_disparity_col)
    d_min_row = np.full((rows.size, cols.size), min_disparity_row)
    d_max_row = np.full((rows.size, cols.size), max_disparity_row)

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
def zeros_cost_volumes(
    rows,
    cols,
    min_disparity_row,
    max_disparity_row,
    min_disparity_col,
    max_disparity_col,
    type_measure,
    subpixel,
):
    """Create a cost_volumes full of zeros."""
    number_of_disparity_col = int((max_disparity_col - min_disparity_col) * subpixel + 1)
    number_of_disparity_row = int((max_disparity_row - min_disparity_row) * subpixel + 1)

    data = np.zeros((rows.size, cols.size, number_of_disparity_col, number_of_disparity_row))
    attrs = {
        "col_disparity_source": [min_disparity_col, max_disparity_col],
        "row_disparity_source": [min_disparity_row, max_disparity_row],
        "col_to_compute": 1,
        "sampling_interval": 1,
        "type_measure": type_measure,
        "step": [1, 1],
        "subpixel": subpixel,
    }

    return PandoraMatchingCostMethods.allocate_cost_volumes(
        attrs,
        rows,
        cols,
        np.linspace(min_disparity_row, max_disparity_row, number_of_disparity_row),
        np.linspace(min_disparity_col, max_disparity_col, number_of_disparity_col),
        data,
    )


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
def disp_map(invalid_disparity, rows, cols):
    """Fake disparity maps with alternating values."""
    row = np.full(rows.size * cols.size, 4.0)
    row[::2] = 5
    col = np.full(rows.size * cols.size, 0.0)
    col[::2] = 1
    return xr.Dataset(
        {
            "row_map": (["row", "col"], row.reshape((rows.size, cols.size))),
            "col_map": (["row", "col"], col.reshape((rows.size, cols.size))),
        },
        coords={
            "row": rows,
            "col": cols,
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
