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

"""
Test create a criteria xarray.Dataset.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import xarray as xr

from pandora2d import criteria
from pandora2d.constants import Criteria


@pytest.fixture()
def grid_min_col(image):
    return image["col_disparity"].sel(band_disp="min")


@pytest.fixture()
def grid_max_col(image):
    return image["col_disparity"].sel(band_disp="max")


@pytest.fixture()
def grid_min_row(image):
    return image["row_disparity"].sel(band_disp="min")


@pytest.fixture()
def grid_max_row(image):
    return image["row_disparity"].sel(band_disp="max")


def test_homogeneous_grids(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row):
    """With uniform grids"""
    make_criteria_copy = criteria_dataarray.copy(deep=True)
    criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

    xr.testing.assert_equal(criteria_dataarray, make_criteria_copy)


def test_variable_col_disparity(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size):
    """With variable column disparity grids"""
    _, col = img_size
    nb_col_set = int(col / 2)
    grid_min_col[:, :nb_col_set] = criteria_dataarray.coords["disp_col"].data[1]
    grid_max_col[:, nb_col_set:] = criteria_dataarray.coords["disp_col"].data[-2]

    criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

    assert np.all(criteria_dataarray.data[:, :nb_col_set, :, 0] == Criteria.P2D_DISPARITY_UNPROCESSED)
    assert np.all(criteria_dataarray.data[:, nb_col_set:, :, 0] == Criteria.VALID)
    assert np.all(criteria_dataarray.data[:, nb_col_set:, :, -1] == Criteria.P2D_DISPARITY_UNPROCESSED)
    assert np.all(criteria_dataarray.data[:, :nb_col_set, :, -1] == Criteria.VALID)


def test_variable_row_disparity(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row, img_size):
    """With variable row disparity grids"""
    row, _ = img_size
    nb_row_set = int(row / 2)
    grid_min_row[:nb_row_set, :] = criteria_dataarray.coords["disp_row"].data[1]
    grid_max_row[nb_row_set:, :] = criteria_dataarray.coords["disp_row"].data[-2]

    criteria.set_unprocessed_disp(criteria_dataarray, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

    assert np.all(criteria_dataarray.data[:nb_row_set, :, 0, :] == Criteria.P2D_DISPARITY_UNPROCESSED)
    assert np.all(criteria_dataarray.data[nb_row_set:, :, 0, :] == Criteria.VALID)
    assert np.all(criteria_dataarray.data[nb_row_set:, :, -1, :] == Criteria.P2D_DISPARITY_UNPROCESSED)
    assert np.all(criteria_dataarray.data[:nb_row_set, :, -1, :] == Criteria.VALID)
