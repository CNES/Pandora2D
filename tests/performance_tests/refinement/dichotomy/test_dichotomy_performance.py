# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

# Make pylint happy with fixtures:
# pylint: disable=too-many-arguments,too-many-positional-arguments

from pathlib import Path

import numpy as np
import pytest
import rasterio

_ZNCC_MC_METHODS = [
    ("zncc-optim-1", "float64"),
    ("zncc-optim-2", "float64"),
    ("zncc_python", "float32"),
]


def _thresholds_for_mc_method(mc_method, thresholds, thresholds_optim_2):
    """
    Select the (row, col) thresholds calibrated for the given matching cost method.

    zncc-optim-2 accumulates floating point sums in a different order than zncc-optim-1, which makes
    its disparity maps diverge from Medicis by about 1e-05 to 6e-05 in columns when subpix is 1.
    Sharing a single threshold would force zncc-optim-1 to inherit that tolerance, and thus lose the
    guarantee that it is at least as accurate as Medicis.
    """

    return thresholds_optim_2 if mc_method == "zncc-optim-2" else thresholds


class TestComparisonMedicis:
    """
    Test that pandora2d mean errors are smaller
    than those of medicis plus a threshold given as a parameter.

    Difference between medicis and pandora2d disparity maps may be linked to the difference
    in the interpolation method used between the two tools when the subpix is greater than 1.
    As pandora2d use the scipy zoom method (spline interpolation),
    medicis use the same interpolation method as the one used for the dichotomy loop (sinc).

    When the threshold is 0, pandora2d is at least as effective as medicis.
    When the threshold is > 0, the mean error of medicis
    is better than the one of pandora2d by about the value of the threshold.

    Each scenario carries two (row, col) threshold couples: `thresholds`, shared by zncc-optim-1 and
    zncc_python, and `thresholds_optim_2`, calibrated for zncc-optim-2.
    """

    def compute_mean_errors(
        self, run_pipeline, remove_edges, cfg_dichotomy, medicis_maps_path, row_shift, col_shift
    ) -> tuple[float, float, float, float]:
        """
        Compute mean errors of medicis and pandora2d disparity maps
        """

        # Run pandora2D pipeline
        run_pipeline(cfg_dichotomy)

        output_path = Path(cfg_dichotomy["output"]["path"])

        # Get pandora2d disparity maps
        with rasterio.open(output_path / "disparity_map" / "row_map.tif") as src:
            row_map_pandora2d = src.read(1)
        with rasterio.open(output_path / "disparity_map" / "col_map.tif") as src:
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

        # Compute mean error between column disparities and real column shift
        mean_error_pandora2d_col = np.nanmean(abs(col_map_pandora2d - col_shift))
        mean_error_medicis_col = np.nanmean(abs(col_map_medicis - col_shift))

        # Compute mean error between row disparities and real row shift
        mean_error_pandora2d_row = np.nanmean(abs(row_map_pandora2d - row_shift))
        mean_error_medicis_row = np.nanmean(abs(row_map_medicis - row_shift))

        # Same float32 or float64 type as Medicis
        mean_error_pandora2d_row = mean_error_pandora2d_row.astype(mean_error_medicis_row)
        mean_error_pandora2d_col = mean_error_pandora2d_col.astype(mean_error_medicis_col)

        return mean_error_pandora2d_row, mean_error_pandora2d_col, mean_error_medicis_row, mean_error_medicis_col

    @pytest.fixture()
    def cfg_dichotomy(self, shift_path, subpix, mc_method, precision, dicho_method, filter_method, tmp_path):
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
                    "matching_cost_method": mc_method,
                    "window_size": 65,
                    "step": [1, 1],
                    "subpix": subpix,
                    "spline_order": 3,
                    "float_precision": precision,
                },  # we use spline_order=3 to get better results when subpix is different from 1
                "disparity": {"disparity_method": "wta", "invalid_disparity": np.nan},
                "refinement": {
                    "refinement_method": dicho_method,
                    "iterations": 9,
                    "filter": {"method": filter_method},
                },
            },
            "output": {"path": str(tmp_path)},
        }

    @pytest.mark.parametrize(("mc_method", "precision"), _ZNCC_MC_METHODS)
    @pytest.mark.parametrize(
        ("dicho_method", "filter_method"),
        [
            ("dichotomy_python", "bicubic"),
            ("dichotomy_python", "bicubic_python"),
            ("dichotomy", "bicubic"),
        ],
    )
    @pytest.mark.parametrize(
        [
            "img_path",
            "subpix",
            "medicis_method_path",
            "row_shift",
            "col_shift",
            "thresholds",
            "thresholds_optim_2",
        ],
        [
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                1,
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.5,
                (0.00001, 0.0),
                (0.0, 0.0),
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                1,
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.5,
                (0.0, 0.0),
                (0.0000058, 0.000059),
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.25/",
                1,
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.25,
                (0.00001, 0.0),
                (0.0, 0.000038),
                id="T19KER (Calama, Chile) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.25/",
                1,
                "zncc_dicho_nappe_bco/gri_zncc_dicho_nappe_bco_",
                0.0,
                0.25,
                (0.0, 0.0),
                (0.0, 0.0000072),
                id="T50JML (Perth, Australia) shifted of 0.25 in columns with bicubic, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                -0.25,
                (0.0014, 0.0),
                (0.0014, 0.0),
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                -0.25,
                (0.0034, 0.0024),
                (0.0034, 0.0024),
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                0.5,
                (0.0026, 0.0021),
                (0.0025, 0.0021),
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.0,
                0.5,
                (0.0033, 0.0016),
                (0.0033, 0.0015),
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.25c+0.25/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.25,
                0.25,
                (0.0, 0.0),
                (0.0, 0.0),
                id="T19KER (Calama, Chile) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.25c+0.25/",
                4,
                "zncc_dicho_nappe_surech_bco/gri_zncc_dicho_nappe_surech_bco_",
                0.25,
                0.25,
                (0.0042, 0.0048),
                (0.0042, 0.0048),
                id="T50JML (Perth, Australia) shifted of 0.25 in col and in rows with bicubic, 9 iter and subpix=4",
            ),
        ],
    )
    def test_pandora2d_medicis_dichotomy_bicubic(
        self,
        run_pipeline,
        remove_edges,
        cfg_dichotomy,
        medicis_maps_path,
        row_shift,
        col_shift,
        thresholds,
        thresholds_optim_2,
    ):
        """
        Tests that the pandora2d disparity maps after using the dichotomy are similar to those obtained with Medicis
        with bicubic filter.
        """

        mc_method = cfg_dichotomy["pipeline"]["matching_cost"]["matching_cost_method"]
        row_map_threshold, col_map_threshold = _thresholds_for_mc_method(mc_method, thresholds, thresholds_optim_2)

        mean_error_pandora2d_row, mean_error_pandora2d_col, mean_error_medicis_row, mean_error_medicis_col = (
            self.compute_mean_errors(run_pipeline, remove_edges, cfg_dichotomy, medicis_maps_path, row_shift, col_shift)
        )

        assert mean_error_pandora2d_col <= mean_error_medicis_col + col_map_threshold
        assert mean_error_pandora2d_row <= mean_error_medicis_row + row_map_threshold

    @pytest.mark.parametrize(
        [
            "img_path",
            "subpix",
            "medicis_method_path",
            "row_shift",
            "col_shift",
            "thresholds",
            "thresholds_optim_2",
        ],
        [
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                1,
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.5,
                (0.0001, 0.0001),
                (0.0, 0.0),
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with sinc_python, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                1,
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.5,
                (0.0, 0.0),
                (0.0, 0.000053),
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with sinc_python, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.25/",
                1,
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.25,
                (0.0, 0.0),
                (0.0, 0.000053),
                id="T19KER (Calama, Chile) shifted of 0.25 in columns with sinc_python, 9 iter and subpix=1",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.25/",
                1,
                "zncc_dicho_nappe_sinc/gri_zncc_dicho_nappe_sinc_",
                0.0,
                0.25,
                (0.00001, 0.00001),
                (0.0, 0.000018),
                id="T50JML (Perth, Australia) shifted of 0.25 in columns with sinc_python, 9 iter and subpix=1",
            ),
            pytest.param(
                "T19KER/r+0.00c-0.25/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                -0.25,
                (0.0025, 0.0097),
                (0.0023, 0.0097),
                id="T19KER (Calama, Chile) shifted of -0.25 in columns with sinc_python, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c-0.25/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                -0.25,
                (0.004, 0.0045),
                (0.0039, 0.0045),
                id="T50JML (Perth, Australia) shifted of -0.25 in columns with sinc_python, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.00c+0.50/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                0.5,
                (0.0028, 0.0036),
                (0.0026, 0.0037),
                id="T19KER (Calama, Chile) shifted of 0.5 in columns with sinc_python, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.00c+0.50/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.0,
                0.5,
                (0.0028, 0.003),
                (0.0028, 0.0028),
                id="T50JML (Perth, Australia) shifted of 0.5 in columns with sinc_python, 9 iter and subpix=4",
            ),
            pytest.param(
                "T19KER/r+0.25c+0.25/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.25,
                0.25,
                (0.0074, 0.0065),
                (0.0077, 0.0066),
                id="T19KER (Calama, Chile) shifted of 0.25 in col and in rows with sinc_python, 9 iter and subpix=4",
            ),
            pytest.param(
                "T50JML/r+0.25c+0.25/",
                4,
                "zncc_dicho_nappe_surech_sinc/gri_zncc_dicho_nappe_surech_sinc_",
                0.25,
                0.25,
                (0.0032, 0.0043),
                (0.0033, 0.0043),
                id="T50JML (Perth, Australia) shifted of 0.25 in col and in rows with sinc_python, 9 iter and subpix=4",
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("dicho_method", "filter_method"),
        [
            ("dichotomy_python", "sinc"),
            ("dichotomy_python", "sinc_python"),
            ("dichotomy", "sinc"),
        ],
    )
    @pytest.mark.parametrize(("mc_method", "precision"), _ZNCC_MC_METHODS)
    def test_pandora2d_medicis_dichotomy_sinc(
        self,
        run_pipeline,
        remove_edges,
        cfg_dichotomy,
        medicis_maps_path,
        row_shift,
        col_shift,
        thresholds,
        thresholds_optim_2,
    ):
        """
        Tests that the pandora2d disparity maps after using the dichotomy are similar to those obtained with Medicis
        with sinc_python filter.
        """

        mc_method = cfg_dichotomy["pipeline"]["matching_cost"]["matching_cost_method"]
        row_map_threshold, col_map_threshold = _thresholds_for_mc_method(mc_method, thresholds, thresholds_optim_2)

        mean_error_pandora2d_row, mean_error_pandora2d_col, mean_error_medicis_row, mean_error_medicis_col = (
            self.compute_mean_errors(run_pipeline, remove_edges, cfg_dichotomy, medicis_maps_path, row_shift, col_shift)
        )

        assert mean_error_pandora2d_col <= mean_error_medicis_col + col_map_threshold
        assert mean_error_pandora2d_row <= mean_error_medicis_row + row_map_threshold
