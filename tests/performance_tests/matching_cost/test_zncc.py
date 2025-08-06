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

"""Tests related to ZNCC."""

# pylint: disable=redefined-outer-name

import json
from copy import deepcopy

import numpy as np
import pytest
import rasterio


import pandora2d


def read_result(config_path, result_path):
    """Read data from result_path relatively to config_path."""
    with config_path.open("r") as cfg_file:
        cfg = json.load(cfg_file)
    output_path = (config_path.parent / cfg["output"]["path"]).resolve()

    # Get pandora2d disparity maps
    with rasterio.open(output_path / result_path) as src:
        return src.read(1)


def compute_mean_error(data: np.typing.NDArray, expected: np.floating) -> np.floating:
    """
    Compute mean error of data.
    """

    # Compute mean error between data and expected
    return np.nanmean(abs(data - expected))


def save_config(configuration, output_dir):
    """Save configuration with name `config.json` in `output_dir`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file_:
        json.dump(configuration, file_, indent=2)
    return config_path


@pytest.fixture()
def configuration(shift_path, subpix):
    """
    Make user configuration
    """

    return {
        "input": {
            "left": {"nodata": -9999, "img": str(shift_path / "left.tif")},
            "right": {"nodata": -9999, "img": str(shift_path / "right.tif")},
            "col_disparity": {"init": 0, "range": 3},
            "row_disparity": {"init": 0, "range": 3},
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc_python",
                "window_size": 65,
                "step": [1, 1],
                "subpix": subpix,
                "spline_order": 3,
                "float_precision": "float32",
            },  # we use spline_order=3 to get better results when subpix is different from 1
            "disparity": {"disparity_method": "wta", "invalid_disparity": np.nan},
        },
        "output": {"path": "./output"},
    }


@pytest.mark.parametrize(
    [
        "img_path",
        "subpix",
        "row_shift",
        "col_shift",
    ],
    [
        pytest.param(
            "T19KER/r+0.00c+0.50/",
            1,
            0.0,
            0.5,
            id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
        ),
        pytest.param(
            "T50JML/r+0.00c+0.50/",
            1,
            0.0,
            0.5,
            id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
        ),
        pytest.param(
            "T19KER/r+0.00c+0.25/",
            1,
            0.0,
            0.25,
            id="T19KER (Calama, Chile) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
        ),
        pytest.param(
            "T50JML/r+0.00c+0.25/",
            1,
            0.0,
            0.25,
            id="T50JML (Perth, Australia) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
        ),
        pytest.param(
            "T19KER/r+0.00c-0.25/",
            4,
            0.0,
            -0.25,
            id="T19KER (Calama, Chile) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
        ),
        pytest.param(
            "T50JML/r+0.00c-0.25/",
            4,
            0.0,
            -0.25,
            id="T50JML (Perth, Australia) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
        ),
        pytest.param(
            "T19KER/r+0.00c+0.50/",
            4,
            0.0,
            0.5,
            id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
        ),
        pytest.param(
            "T50JML/r+0.00c+0.50/",
            4,
            0.0,
            0.5,
            id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
            marks=pytest.mark.xfail(reason="To be investigated."),
        ),
        pytest.param(
            "T19KER/r+0.25c+0.25/",
            4,
            0.25,
            0.25,
            id="T19KER (Calama, Chile) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
        ),
        pytest.param(
            "T50JML/r+0.25c+0.25/",
            4,
            0.25,
            0.25,
            id="T50JML (Perth, Australia) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
        ),
    ],
)
@pytest.mark.parametrize(
    "cpp_float_precision",
    ["float64", "float32"],
)
def test_compare_znccs(
    tmp_path,
    configuration,
    row_shift,
    col_shift,
    cpp_float_precision,
):
    """
    Tests that the pandora2d disparity maps from ZNCC in Python or C++ are similar.
    """

    zncc_python_config_path = save_config(configuration, tmp_path / "python")

    zncc_cpp_config = deepcopy(configuration)
    zncc_cpp_config["pipeline"]["matching_cost"]["matching_cost_method"] = "zncc"
    zncc_cpp_config["pipeline"]["matching_cost"]["float_precision"] = cpp_float_precision
    zncc_cpp_config_path = save_config(zncc_cpp_config, tmp_path / "cpp")

    pandora2d.main(zncc_python_config_path, verbose=False)
    pandora2d.main(zncc_cpp_config_path, verbose=False)

    cpp_row_map = read_result(zncc_cpp_config_path, "disparity_map/row_map.tif")
    python_row_map = read_result(zncc_python_config_path, "disparity_map/row_map.tif")
    mean_row_error_python = compute_mean_error(cpp_row_map, row_shift)
    mean_row_error_cpp = compute_mean_error(python_row_map, row_shift)

    cpp_col_map = read_result(zncc_cpp_config_path, "disparity_map/col_map.tif")
    python_col_map = read_result(zncc_python_config_path, "disparity_map/col_map.tif")
    mean_col_error_python = compute_mean_error(cpp_col_map, col_shift)
    mean_col_error_cpp = compute_mean_error(python_col_map, col_shift)

    np.testing.assert_array_equal(cpp_row_map, python_row_map)
    np.testing.assert_array_equal(cpp_col_map, python_col_map)
    assert mean_row_error_cpp == pytest.approx(mean_row_error_python)
    assert mean_col_error_cpp == pytest.approx(mean_col_error_python)
