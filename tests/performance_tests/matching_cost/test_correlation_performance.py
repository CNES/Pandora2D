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
Test the correlation methods performance.
"""

# pylint: disable=redefined-outer-name, too-few-public-methods
import json
from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
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
def configuration(shift_path, method, subpix, tmp_path):
    """
    Make user configuration for mutual information computation
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
                "matching_cost_method": method,
                "window_size": 65,
                "step": [1, 1],
                "subpix": subpix,
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": np.nan},
        },
        "output": {
            "path": str(tmp_path / "vs_medicis_output"),
        },
    }


class TestComparisonMedicis:
    """
    Test that pandora2d disparity maps are equal or close to the medicis ones
    when matching cost is used with mutual information and zncc methods.
    """

    @pytest.mark.parametrize(
        [
            "img_path",
            "method",
            "subpix",
            "medicis_method_path",
        ],
        [
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                "mutual_information",
                2,
                "mi/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with subpix=2, mutual_information",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                "mutual_information",
                2,
                "mi/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with subpix=2, mutual_information",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                "mutual_information",
                4,
                "mi/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with subpix=4, mutual_information",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                "mutual_information",
                4,
                "mi/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with subpix=4, mutual_information",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                "zncc",
                2,
                "zncc/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with subpix=2, zncc",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                "zncc",
                2,
                "zncc/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with subpix=2, zncc",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                "zncc",
                4,
                "zncc/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with subpix=4, zncc",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                "zncc",
                4,
                "zncc/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with subpix=4, zncc",
            ),
        ],
    )
    def test_comparaison_pandora2d_medicis(self, run_pipeline, remove_edges, configuration, medicis_maps_path):
        """
        Compare medicis and pandora2d disparity maps
        """
        output_dir = Path(configuration["output"]["path"])
        # Run pandora2D pipeline
        run_pipeline(configuration)

        # Get pandora2d disparity maps
        with rasterio.open(output_dir / "disparity_map" / "row_map.tif") as src:
            row_map_pandora2d = src.read(1)
        with rasterio.open(output_dir / "disparity_map" / "col_map.tif") as src:
            col_map_pandora2d = src.read(1)

        # Get medicis disparity maps
        with rasterio.open(str(medicis_maps_path) + "row_disp.tif") as src:
            row_map_medicis = src.read(1)
        with rasterio.open(str(medicis_maps_path) + "col_disp.tif") as src:
            col_map_medicis = src.read(1)

        # Remove medicis edges on both pandora2d and medicis disparity maps
        # in order to compare the same sample of points.
        row_map_medicis, row_map_pandora2d = remove_edges(row_map_medicis, row_map_pandora2d)
        col_map_medicis, col_map_pandora2d = remove_edges(col_map_medicis, col_map_pandora2d)

        np.testing.assert_array_equal(row_map_medicis, row_map_pandora2d)
        np.testing.assert_array_equal(col_map_medicis, col_map_pandora2d)


class TestComparisonZncc:
    """
    Test that pandora2d disparity maps are equal or close when using cpp or python zncc implementations.
    """

    @pytest.fixture()
    def method(self):
        return "zncc_python"

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
        self,
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
