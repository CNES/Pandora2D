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
Test the refinement.dichotomy pipeline.
"""

from typing import Tuple
from pathlib import Path

import pytest

import numpy as np
import rasterio

from numpy.typing import NDArray


class TestComparisonMedicis:
    """
    Tests that the pandora2d disparity maps after using the dichotomy
    are similar to those obtained with Medicis.
    """

    def remove_edges(
        self, medicis_map: NDArray[np.floating], pandora2d_map: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get reduced disparity maps after removing medicis edges full of nans (greater than pandora2d edges)
        on both pandora2d and medicis disparity maps.
        """

        # Gets coordinates for which medicis col_map is different from nan
        # i.e. points that are not within the edges
        non_nan_row_indexes, non_nan_col_indexes = np.where(~np.isnan(medicis_map))

        # Remove medicis edges
        medicis_map = medicis_map[
            non_nan_row_indexes[0] : non_nan_row_indexes[-1] + 1, non_nan_col_indexes[0] : non_nan_col_indexes[-1] + 1
        ]

        # Remove pandora2d edges to get the same points as the ones in medicis disparity maps
        pandora2d_map = pandora2d_map[
            non_nan_row_indexes[0] : non_nan_row_indexes[-1] + 1, non_nan_col_indexes[0] : non_nan_col_indexes[-1] + 1
        ]

        return medicis_map, pandora2d_map

    @pytest.fixture()
    def data_path(self):
        """
        Return path to get left and right images and medicis data
        """
        return Path("tests/performance_tests/refinement/dichotomy/data_medicis/")

    @pytest.fixture()
    def shift_path(self, data_path, img_path):
        """
        Return path to get left and right images and medicis data
        """
        return data_path / img_path

    @pytest.fixture()
    def medicis_maps_path(self, shift_path, medicis_method_path):
        """
        Return path to get medicis data
        """
        return shift_path / medicis_method_path

    @pytest.fixture()
    def cfg_dichotomy(self, shift_path, subpix, filter_method):
        """
        Make user configuration for dichotomy loop
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
                    "matching_cost_method": "zncc",
                    "window_size": 65,
                    "step": [1, 1],
                    "subpix": subpix,
                    "spline_order": 3,
                },  # we use spline_order=3 to get better results when subpix is different from 1
                "disparity": {"disparity_method": "wta", "invalid_disparity": np.nan},
                "refinement": {"refinement_method": "dichotomy", "iterations": 9, "filter": {"method": filter_method}},
            },
        }

    @pytest.mark.parametrize(
        [
            "img_path",
            "subpix",
            "filter_method",
            "medicis_method_path",
            "row_shift",
            "col_shift",
            "row_map_threshold",
            "col_map_threshold",
        ],
        [
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                1,
                "bicubic",
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.5,
                0.00001,
                0.0,
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                1,
                "bicubic",
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.5,
                0.0,
                0.0,
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.25/",
                1,
                "bicubic",
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.25,
                0.00001,
                0.0,
                id="T19KER (Calama, Chile) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.25/",
                1,
                "bicubic",
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.25,
                0.0,
                0.0,
                id="T50JML (Perth, Australia) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                -0.25,
                0.002,
                0.0,
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                -0.25,
                0.004,
                0.003,
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                0.5,
                0.004,
                0.003,
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                0.5,
                0.004,
                0.002,
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.25c+0.25/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.25,
                0.25,
                0.0,
                0.0,
                id="T19KER (Calama, Chile) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.25c+0.25/",
                4,
                "bicubic",
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.25,
                0.25,
                0.005,
                0.005,
                id="T50JML (Perth, Australia) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                1,
                "sinc",
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.5,
                0.0001,
                0.0001,
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with sinc, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                1,
                "sinc",
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.5,
                0.0,
                0.0,
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with sinc, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.25/",
                1,
                "sinc",
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.25,
                0.0,
                0.0,
                id="T19KER (Calama, Chile) shifted of 0.25 in columns with sinc, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.25/",
                1,
                "sinc",
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.25,
                0.00001,
                0.00001,
                id="T50JML (Perth, Australia) shifted of 0.25 in columns with sinc, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                -0.25,
                0.003,
                0.01,
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with sinc, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                -0.25,
                0.004,
                0.005,
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with sinc, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                0.5,
                0.003,
                0.004,
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with sinc, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                0.5,
                0.003,
                0.003,
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with sinc, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.25c+0.25/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.25,
                0.25,
                0.01,
                0.007,
                id="T19KER (Calama, Chile) shifted of 0.25 in col and in rows with sinc, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.25c+0.25/",
                4,
                "sinc",
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.25,
                0.25,
                0.004,
                0.005,
                id="T50JML (Perth, Australia) shifted of 0.25 in col and in rows with sinc, 9 iter and subpix=4",
            ),
        ],
    )
    def test_pandora2d_medicis_dichotomy(
        self, run_pipeline, cfg_dichotomy, medicis_maps_path, row_shift, col_shift, row_map_threshold, col_map_threshold
    ):
        """
        Test that pandora2d mean errors are smaller
        than those of medicis plus a threshold given as a parameter.

        Difference between medicis and pandora2d disparity maps may be linked to the difference
        in the interpolation method used between the two tools when the subpix is greater than 1.
        As pandora2d use the scipy zoom method (spline interpolation),
        medicis use the same interpolation method as the one used for the dichotomy loop (bicubic or sinc).

        When the threshold is 0, pandora2d is at least as effective as medicis.
        When the threshold is > 0, the mean error of medicis
        is better than the one of pandora2d by about the value of the threshold.
        """

        # Run pandora2D pipeline
        run_dir = run_pipeline(cfg_dichotomy)

        # Get pandora2d disparity maps
        with rasterio.open(run_dir / "output" / "row_disparity.tif") as src:
            row_map_pandora2d = src.read(1)
        with rasterio.open(run_dir / "output" / "columns_disparity.tif") as src:
            col_map_pandora2d = src.read(1)

        # Get medicis disparity maps
        with rasterio.open(str(medicis_maps_path) + "row_disp.tif") as src:
            row_map_medicis = src.read(1)
        with rasterio.open(str(medicis_maps_path) + "col_disp.tif") as src:
            col_map_medicis = src.read(1)

        # Remove medicis edges on both pandora2d and medicis disparity maps
        # in order to compare the same sample of points.
        row_map_medicis, row_map_pandora2d = self.remove_edges(row_map_medicis, row_map_pandora2d)
        col_map_medicis, col_map_pandora2d = self.remove_edges(col_map_medicis, col_map_pandora2d)

        # Compute mean error between column disparities and real column shift
        mean_error_pandora2d_col = np.nanmean(abs(col_map_pandora2d - col_shift))
        mean_error_medicis_col = np.nanmean(abs(col_map_medicis - col_shift))

        # Compute mean error between row disparities and real row shift
        mean_error_pandora2d_row = np.nanmean(abs(row_map_pandora2d - row_shift))
        mean_error_medicis_row = np.nanmean(abs(row_map_medicis - row_shift))

        assert mean_error_pandora2d_col <= mean_error_medicis_col + col_map_threshold
        assert mean_error_pandora2d_row <= mean_error_medicis_row + row_map_threshold
