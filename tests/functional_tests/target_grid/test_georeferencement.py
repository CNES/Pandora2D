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


@pytest.fixture()
def configuration(left_path, right_path, correct_pipeline_without_refinement, step):
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
    }


@pytest.mark.parametrize(
    [
        "step",
        "bottom_right_corner_indexes",  # Use transform convention: (col, row)
    ],
    [
        pytest.param([1, 1], (9, 9), id="No step"),  # Disp map corner match ROI corner
        pytest.param([2, 3], (9, 8), id="Step < image"),  # Disp map corner match ROI corner
        pytest.param([11, 11], (0, 0), id="Step > image"),  # Disp map corner match ROI corner
    ],
)
@pytest.mark.parametrize("output_file", ["columns_disparity.tif", "row_disparity.tif", "correlation_score.tif"])
def test_georeferencement(
    run_pipeline,
    configuration,
    crs,
    transform,
    bottom_right_corner_indexes,
    output_file,
):
    """Test that top left and bottom right corners are well georeferenced."""
    run_dir = run_pipeline(configuration)

    output = rasterio.open(run_dir / "output" / output_file)
    bottom_right_disparity_indexes = output.width - 1, output.height - 1

    assert output.crs == crs
    # assert that new georeferencement origin correspond to upper left corner of the ROI:
    upper_left_corner_indexes = (0, 0)
    assert output.transform * (0, 0) == transform * upper_left_corner_indexes
    assert output.transform * bottom_right_disparity_indexes == transform * bottom_right_corner_indexes


@pytest.fixture()
def configuration_with_roi(configuration, roi):
    configuration["ROI"] = roi
    return configuration


@pytest.mark.parametrize(
    [
        "roi",
        "step",
        "bottom_right_corner_indexes",  # Use transform convention: (col, row)
    ],
    [
        pytest.param(
            {"col": {"first": 3, "last": 7}, "row": {"first": 5, "last": 8}}, [1, 1], (7, 8), id="No step"
        ),  # Disp map corner match ROI corner
        pytest.param(
            {"col": {"first": 3, "last": 7}, "row": {"first": 5, "last": 8}},
            [2, 3],
            (6, 7),
            id="Step < ROI size",
        ),  # Disp map corner is inside ROI
        pytest.param(
            {"col": {"first": 3, "last": 7}, "row": {"first": 5, "last": 8}},
            [4, 5],
            (3, 5),
            id="Step == ROI size",
        ),  # Only one pixel at ROI origin
        pytest.param(
            {"col": {"first": 3, "last": 7}, "row": {"first": 5, "last": 8}},
            [5, 6],
            (3, 5),
            id="Step > ROI size",
        ),  # Only one pixel at ROI origin
        pytest.param(
            {"col": {"first": 3, "last": 3}, "row": {"first": 5, "last": 5}},
            [1, 1],
            (3, 5),
            id="1px ROI - No step",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 3}, "row": {"first": 5, "last": 5}},
            [5, 6],
            (3, 5),
            id="1px ROI - Step",
        ),
    ],
)
@pytest.mark.parametrize("output_file", ["columns_disparity.tif", "row_disparity.tif", "correlation_score.tif"])
def test_roi_georeferencement(
    run_pipeline,
    configuration_with_roi,
    crs,
    transform,
    bottom_right_corner_indexes,
    output_file,
):
    """Test that top left and bottom right corners are well georeferenced."""
    run_dir = run_pipeline(configuration_with_roi)

    output = rasterio.open(run_dir / "output" / output_file)
    bottom_right_disparity_indexes = output.width - 1, output.height - 1

    assert output.crs == crs
    # assert that new georeferencement origin correspond to upper left corner of the ROI:
    upper_left_corner_indexes = (
        configuration_with_roi["ROI"]["col"]["first"],
        configuration_with_roi["ROI"]["row"]["first"],
    )
    assert output.transform * (0, 0) == transform * upper_left_corner_indexes
    assert output.transform * bottom_right_disparity_indexes == transform * bottom_right_corner_indexes
