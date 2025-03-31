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
#
"""Module with global test fixtures."""

# pylint: disable=redefined-outer-name

import rasterio
import xarray as xr
import numpy as np
import pytest
from pandora.common import write_data_array


@pytest.fixture()
def correct_pipeline_with_optical_flow():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "optical_flow"},
        }
    }


@pytest.fixture()
def correct_pipeline_with_dichotomy_python():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "dichotomy_python", "iterations": 2, "filter": {"method": "bicubic"}},
        }
    }


@pytest.fixture()
def correct_pipeline_with_dichotomy_cpp():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "dichotomy", "iterations": 2, "filter": {"method": "bicubic"}},
        }
    }


@pytest.fixture()
def row_disparity():
    return {"init": 0, "range": 2}


@pytest.fixture()
def col_disparity():
    return {"init": 2, "range": 1}


@pytest.fixture()
def correct_input_for_functional_tests(left_img_path, right_img_path, col_disparity, row_disparity):
    return {
        "input": {
            "left": {"img": str(left_img_path), "nodata": "NaN", "mask": None},
            "right": {"img": str(right_img_path), "nodata": "NaN", "mask": None},
            "col_disparity": col_disparity,
            "row_disparity": row_disparity,
        }
    }


@pytest.fixture()
def left_data():
    """Create left data."""
    # Create array of shape (10,10):
    # [[ 0,  1,  2, ...,  0],
    #  [ 1,  2,  3, ...,  10],
    #  [...],
    #  [8,  9,  10, ...,  17]]
    #  [9, 10,  11, ...,  18]]
    return xr.DataArray(
        data=np.arange(10) + np.arange(10).reshape(-1, 1),
        dims=("row", "col"),
        coords={"row": np.arange(10), "col": np.arange(10)},
    )


@pytest.fixture()
def right_data(left_data):
    return left_data + 1


@pytest.fixture()
def transform():
    return rasterio.Affine(0.5, 0.0, 573083.5, 0.0, -0.5, 4825333.5)


@pytest.fixture()
def crs():
    return rasterio.crs.CRS.from_epsg(32631)


@pytest.fixture()
def left_path(tmp_path, left_data, crs, transform):
    """Write left image and return its path."""
    path = tmp_path / "left.tif"
    write_data_array(
        data_array=left_data,
        filename=str(path),
        crs=crs,
        transform=transform,
    )
    return path


@pytest.fixture()
def right_path(tmp_path, right_data, crs, transform):
    """Write right image and return its path."""
    path = tmp_path / "right.tif"
    write_data_array(
        data_array=right_data,
        filename=str(path),
        crs=crs,
        transform=transform,
    )
    return path


@pytest.fixture()
def configuration(left_path, right_path, correct_pipeline_without_refinement, step, tmp_path):
    correct_pipeline_without_refinement["pipeline"]["matching_cost"]["step"] = step
    return {
        "input": {
            "left": {
                "img": str(left_path),
            },
            "right": {
                "img": str(right_path),
            },
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 2},
        },
        **correct_pipeline_without_refinement,
        "output": {"path": str(tmp_path.absolute())},
    }
