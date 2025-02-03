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
Test the refinement.dichotomy pipeline.
"""
from pathlib import Path

import pytest
import numpy as np
import rasterio


class TestComparisonMedicis:
    """
    Test that pandora2d disparity maps are equal or close to the medicis ones
    when matching cost is used with mutual information method.
    """

    @pytest.fixture()
    def cfg_mutual_information(self, shift_path, subpix, tmp_path):
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
                    "matching_cost_method": "mutual_information",
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

    @pytest.mark.parametrize(
        [
            "img_path",
            "subpix",
            "medicis_method_path",
        ],
        [
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                2,
                "mi/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with subpix=2",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                2,
                "mi/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with subpix=2",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                4,
                "mi/gri_resultat_",
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                4,
                "mi/gri_resultat_",
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with subpix=4",
            ),
        ],
    )
    def test_pandora2d_medicis_mutual_information(
        self, run_pipeline, remove_edges, cfg_mutual_information, medicis_maps_path
    ):
        """
        Compute mean errors of medicis and pandora2d disparity maps
        """
        output_dir = Path(cfg_mutual_information["output"]["path"])
        # Run pandora2D pipeline
        run_pipeline(cfg_mutual_information)

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
