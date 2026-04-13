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
Functional tests for configurations with variable initial disparity.
"""

# pylint: disable=redefined-outer-name, too-many-arguments, too-many-positional-arguments

from copy import deepcopy

import numpy as np
import pytest
import rasterio


# /!\ "zncc" currently target "zncc-optim-1"
@pytest.mark.parametrize("matching_cost_method", ["zncc", "zncc-optim-2", "zncc_python", "mutual_information"])
@pytest.mark.parametrize(
    ["make_input_cfg", "pipeline"],
    [
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_without_refinement",
            id="Pipeline with disparity grids",
        ),
        pytest.param(
            {"row_disparity": "same_sized_grid_directory", "col_disparity": "same_sized_grid_directory"},
            "correct_pipeline_without_refinement",
            id="Pipeline with disparity grid directory",
        ),
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_with_dichotomy_python",
            id="Pipeline with disparity grids and dichotomy python",
        ),
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_with_dichotomy_cpp",
            id="Pipeline with disparity grids and dichotomy cpp",
        ),
    ],
    indirect=["make_input_cfg"],
)
def test_disparity_grids(
    run_pipeline, make_input_cfg, pipeline, request, tmp_path, matching_cost_method
):  # pylint: disable=unused-argument
    """
    Description: Test pipeline with disparity grids
    """

    configuration = {
        "input": make_input_cfg,
        "ROI": {"col": {"first": 210, "last": 240}, "row": {"first": 210, "last": 240}},
        **request.getfixturevalue(pipeline),
        **{"output": {"path": str(tmp_path)}},
    }
    configuration["pipeline"]["disparity"]["invalid_disparity"] = np.nan

    run_pipeline(configuration)

    with rasterio.open(tmp_path / "disparity_map" / "row_map.tif") as src:
        row_map = src.read(1)
    with rasterio.open(tmp_path / "disparity_map" / "col_map.tif") as src:
        col_map = src.read(1)

    non_nan_row_map = ~np.isnan(row_map)
    non_nan_col_map = ~np.isnan(col_map)

    # Minimal and maximal disparities corresponding to correct_grid_path fixture
    min_max_disp_row = np.array(
        [
            np.tile([[-3], [-5], [-2]], (375 // 3 + 1, 450))[210:241, 210:241],
            np.tile([[7], [5], [8]], (375 // 3 + 1, 450))[210:241, 210:241],
        ]
    )

    # Minimal and maximal disparities corresponding to second_correct_grid_path fixture
    min_max_disp_col = np.array(
        [
            np.tile([[0, -26, -6]], (375, 450 // 3 + 1))[210:241, 210:241],
            np.tile([[10, -16, 4]], (375, 450 // 3 + 1))[210:241, 210:241],
        ]
    )

    # Checks that the resulting disparities are well within the ranges created from the input disparity grids
    assert np.all(
        (row_map[non_nan_row_map] >= min_max_disp_row[0, ::][non_nan_row_map])
        & (row_map[non_nan_row_map] <= min_max_disp_row[1, ::][non_nan_row_map])
    )
    assert np.all(
        (col_map[non_nan_col_map] >= min_max_disp_col[0, ::][non_nan_col_map])
        & (col_map[non_nan_col_map] <= min_max_disp_col[1, ::][non_nan_col_map])
    )


class TestDirectoryDisparityPipeline:
    """
    Test directory initial disparity pipeline
    """

    @pytest.fixture()
    def configuration(self, input_cfg, pipeline_cfg, roi, tmp_path, request):
        """
        User configuration
        """

        input_cfg = request.getfixturevalue(input_cfg)
        pipeline_cfg = request.getfixturevalue(pipeline_cfg)

        configuration = {
            **input_cfg,
            **pipeline_cfg,
            **{"output": {"path": str(tmp_path)}},
        }

        if roi is not None:
            configuration.update(roi)

        return configuration

    @pytest.fixture()
    def invalid_disparity_first_pipeline(self, configuration, run_pipeline, tmp_path):
        """
        Run first Pandora 2D pipeline to get invalid disparity mask
        """

        # Run first pipeline to create disparity maps
        run_pipeline(configuration)

        with rasterio.open(tmp_path / "disparity_map" / "row_map.tif") as src:
            row_map_first = src.read(1)
        with rasterio.open(tmp_path / "disparity_map" / "col_map.tif") as src:
            col_map_first = src.read(1)

        invalid_row_disparity = row_map_first == configuration["pipeline"]["disparity"]["invalid_disparity"]
        invalid_col_disparity = col_map_first == configuration["pipeline"]["disparity"]["invalid_disparity"]
        invalid_disparity = invalid_row_disparity | invalid_col_disparity

        return invalid_disparity

    @pytest.fixture()
    def second_configuration(self, configuration, tmp_path):
        """
        Create configuration for second pipeline run with initial disparity
        set to the disparity computed in the first run
        """

        second_configuration = deepcopy(configuration)
        # Remove ROI from configuration because it will be reconstructed from initial disparity grids
        second_configuration.pop("ROI", None)
        second_configuration["input"]["row_disparity"]["init"] = str(tmp_path / "disparity_map")
        second_configuration["input"]["col_disparity"]["init"] = str(tmp_path / "disparity_map")

        return second_configuration

    @pytest.mark.parametrize("input_cfg", ["correct_input_cfg", "correct_input_with_left_right_mask"])
    @pytest.mark.parametrize("pipeline_cfg", ["correct_pipeline_with_dichotomy_cpp"])
    @pytest.mark.parametrize("matching_cost_method", ["zncc", "sad"])
    @pytest.mark.parametrize("step", [[1, 1], [3, 10]])
    @pytest.mark.parametrize("subpix", [1, 2])
    @pytest.mark.parametrize("invalid_disparity", [-99, np.nan])
    @pytest.mark.parametrize(
        "roi", [None, {"ROI": {"col": {"first": 10, "last": 100}, "row": {"first": 250, "last": 350}}}]
    )
    def test_directory_initial_disparity(
        self,
        second_configuration,
        invalid_disparity_first_pipeline,
        matching_cost_method,
        step,
        subpix,
        invalid_disparity,
        run_pipeline,
        tmp_path,
    ):  # pylint: disable=unused-argument
        """
        Test execution of a pipeline with variable initial disparity given as a directory.
        """

        run_pipeline(second_configuration)

        with rasterio.open(tmp_path / "disparity_map" / "row_map.tif") as src:
            row_map_second = src.read(1)
        with rasterio.open(tmp_path / "disparity_map" / "col_map.tif") as src:
            col_map_second = src.read(1)
        with rasterio.open(tmp_path / "disparity_map" / "validity.tif") as dataset:
            invalid_init_disp_band = dataset.read(dataset.descriptions.index("P2D_INVALID_INIT_DISPARITY") + 1)
            left_border_band = dataset.read(dataset.descriptions.index("P2D_LEFT_BORDER") + 1)
            partial_validity_band = dataset.read(dataset.descriptions.index("partial_validity_mask") + 1)

        # The invalid initial disparity mask corresponds to pixels that have invalid disparity in the first run
        invalid_init_disp_mask = invalid_init_disp_band == 1

        # We remove from the invalid disparity mask the pixels that are on the left border because
        # the P2D_LEFT_BORDER criterion override other criteria.
        invalid_disparity_mask = invalid_disparity_first_pipeline & (left_border_band == 0)

        # Checking that resulting disparities are not full of nans
        assert not np.all(np.isnan(row_map_second))
        assert not np.all(np.isnan(col_map_second))
        # Check that pixels with invalid initial disparity in the second run
        # are the ones with invalid disparity in the first run
        assert np.array_equal(invalid_init_disp_mask, invalid_disparity_mask)
        # Check that pixels with invalid initial disparity have
        # a value of 1 in the partial validity mask of the second run
        assert np.all(partial_validity_band[invalid_disparity_mask] == 1)
