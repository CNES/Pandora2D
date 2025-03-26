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
Run pandora2d configurations with ROI from end to end.
"""
from pathlib import Path

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import pytest
import rasterio

from pandora.common import write_data_array


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
@pytest.mark.parametrize("output_file", ["col_map.tif", "row_map.tif", "correlation_score.tif"])
def test_georeferencement(
    run_pipeline,
    configuration,
    crs,
    transform,
    bottom_right_corner_indexes,
    output_file,
):
    """Test that top left and bottom right corners are well georeferenced."""
    run_pipeline(configuration)

    output = rasterio.open(Path(configuration["output"]["path"]) / "disparity_map" / output_file)
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
@pytest.mark.parametrize("output_file", ["col_map.tif", "row_map.tif", "correlation_score.tif"])
def test_roi_georeferencement(
    run_pipeline,
    configuration_with_roi,
    crs,
    transform,
    bottom_right_corner_indexes,
    output_file,
):
    """Test that top left and bottom right corners are well georeferenced."""
    run_pipeline(configuration_with_roi)

    output = rasterio.open(Path(configuration_with_roi["output"]["path"]) / "disparity_map" / output_file)
    bottom_right_disparity_indexes = output.width - 1, output.height - 1

    assert output.crs == crs
    # assert that new georeferencement origin correspond to upper left corner of the ROI:
    upper_left_corner_indexes = (
        configuration_with_roi["ROI"]["col"]["first"],
        configuration_with_roi["ROI"]["row"]["first"],
    )
    assert output.transform * (0, 0) == transform * upper_left_corner_indexes
    assert output.transform * bottom_right_disparity_indexes == transform * bottom_right_corner_indexes
