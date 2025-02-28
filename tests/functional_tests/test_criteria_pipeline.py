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
"""
Test that criteria datarray and validity mask are correct on different pandora2d pipeline
"""
# pylint: disable=too-many-lines
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import rasterio

from pandora2d.constants import Criteria
from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d.state_machine import Pandora2DMachine
from pandora2d import run
from pandora2d.check_configuration import check_conf


@pytest.fixture()
def ground_truth_criteria_dataarray(left_img_shape):
    """
    Criteria dataarray ground truth
    for test_criteria_datarray_created_in_state_machine.

    This ground truth is based on the parameters (window_size, disparity) of the
    correct_input_cfg and correct_pipeline_without_refinement fixtures.
    """

    # disp = {"init": 1, "range":2} -> range size = 5
    ground_truth = np.full((left_img_shape[0], left_img_shape[1], 5, 5), Criteria.VALID)

    # Here, window_size=5
    # For disp=-1, 3 first column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:3, :, 0, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, :3, :, 0] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=1, 3 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-3:, :, 2, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -3:, :, 2] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=2, 4 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-4:, :, 3, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -4:, :, 3] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # For disp=3, 5 last column/row are equal to Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-5:, :, 4, :] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -5:, :, 4] = Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE

    # Window_size=5, so the two first and last rows and columns are equal to Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:2, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:, :2, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[-2:, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
    ground_truth[:, -2:, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER

    return ground_truth


def test_criteria_datarray_created_in_cost_volumes(
    correct_input_cfg, correct_pipeline_without_refinement, ground_truth_criteria_dataarray
):
    """
    Test that pandora2d machine contains the criteria dataarray
    """

    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement, "output": {"path": "rala"}}

    pandora2d_machine = Pandora2DMachine()

    checked_cfg = check_conf(configuration, pandora2d_machine)

    img_left, img_right = create_datasets_from_inputs(input_config=checked_cfg["input"])

    dataset_disp_maps, _ = run(pandora2d_machine, img_left, img_right, checked_cfg)

    # Get peak on the edges to add Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE in ground_truth_criteria_dataarray
    row_peak_mask = (
        dataset_disp_maps["row_map"].data
        == correct_input_cfg["input"]["row_disparity"]["init"] - correct_input_cfg["input"]["row_disparity"]["range"]
    ) | (
        dataset_disp_maps["row_map"].data
        == correct_input_cfg["input"]["row_disparity"]["init"] + correct_input_cfg["input"]["row_disparity"]["range"]
    )

    col_peak_mask = (
        dataset_disp_maps["col_map"].data
        == correct_input_cfg["input"]["col_disparity"]["init"] - correct_input_cfg["input"]["col_disparity"]["range"]
    ) | (
        dataset_disp_maps["col_map"].data
        == correct_input_cfg["input"]["col_disparity"]["init"] + correct_input_cfg["input"]["col_disparity"]["range"]
    )

    ground_truth_criteria_dataarray[row_peak_mask | col_peak_mask] |= Criteria.PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE

    # Check that criteria dataarray contains correct criteria
    np.testing.assert_array_equal(pandora2d_machine.cost_volumes["criteria"].data, ground_truth_criteria_dataarray)


def test_validity_mask_saved(correct_input_cfg, correct_pipeline_without_refinement, run_pipeline, tmp_path):
    """
    Test that validity_mask is correctly saved when pandora2d pipeline is executed
    """

    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement, **{"output": {"path": str(tmp_path)}}}

    run_pipeline(configuration)

    validity_mask_path = tmp_path / "disparity_map" / "validity.tif"

    expected_band_names = ("validity_mask", "criteria_1")

    # Check that validity_mask.tif exists
    assert (validity_mask_path).exists()

    # Check that validity_mask.tif contains two bands of type int8
    with rasterio.open(validity_mask_path) as dataset:
        assert dataset.count == 2
        assert dataset.dtypes[0] == "uint8"
        assert dataset.dtypes[1] == "uint8"
        assert dataset.descriptions == expected_band_names
