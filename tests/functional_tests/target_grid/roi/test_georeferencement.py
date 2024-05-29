# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Run pandora2d configurations with ROI from end to end.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import pytest
import xarray as xr
import numpy as np
import rasterio

from pandora.common import write_data_array


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


def test_roi_georeferencement(run_pipeline, left_path, right_path, crs, transform, correct_pipeline_without_refinement):
    """Test that new georeferencement origin correspond to upper left corner of the ROI."""
    configuration = {
        "input": {
            "left": {
                "img": str(left_path),
            },
            "right": {
                "img": str(right_path),
            },
            "col_disparity": [-1, 3],
            "row_disparity": [-1, 3],
        },
        "ROI": {
            "col": {"first": 3, "last": 7},
            "row": {"first": 5, "last": 8},
        },
        **correct_pipeline_without_refinement,
    }

    run_dir = run_pipeline(configuration)

    columns_disparity = rasterio.open(run_dir / "output" / "columns_disparity.tif")

    assert columns_disparity.crs == crs
    # assert that new georeferencement origin correspond to upper left corner of the ROI:
    assert columns_disparity.transform * (0, 0) == transform * (3, 5)
