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
    # For disp=-1, 3 first column/row are equal to Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:3, :, 0, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, :3, :, 0] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

    # For disp=1, 3 last column/row are equal to Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-3:, :, 2, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -3:, :, 2] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

    # For disp=2, 4 last column/row are equal to Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-4:, :, 3, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -4:, :, 3] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

    # For disp=3, 5 last column/row are equal to Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[-5:, :, 4, :] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE
    ground_truth[:, -5:, :, 4] = Criteria.P2D_RIGHT_DISPARITY_OUTSIDE

    # Window_size=5, so the two first and last rows and columns are equal to Criteria.P2D_LEFT_BORDER
    ground_truth[:2, :, :, :] = Criteria.P2D_LEFT_BORDER
    ground_truth[:, :2, :, :] = Criteria.P2D_LEFT_BORDER
    ground_truth[-2:, :, :, :] = Criteria.P2D_LEFT_BORDER
    ground_truth[:, -2:, :, :] = Criteria.P2D_LEFT_BORDER

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

    _, __ = run(pandora2d_machine, img_left, img_right, checked_cfg)

    # Check that criteria dataarray contains correct criteria
    np.testing.assert_array_equal(pandora2d_machine.cost_volumes["criteria"].data, ground_truth_criteria_dataarray)


@pytest.mark.parametrize(
    ["input_cfg", "step", "subpix"],
    [
        pytest.param(
            "correct_input_cfg",
            [1, 1],
            1,
            id="No mask and step=[1,1]",
        ),
        pytest.param(
            "correct_input_with_left_mask",
            [2, 1],
            1,
            id="Left mask and step=[2,1]",
        ),
        pytest.param(
            "correct_input_with_right_mask",
            [1, 3],
            2,
            id="Right mask, step=[1,3] and subpix=2",
        ),
        pytest.param(
            "correct_input_with_left_right_mask",
            [4, 5],
            4,
            id="Left and right masks, step=[4,5] and subpix=4",
        ),
    ],
)
def test_validity_mask_saved(
    input_cfg, step, subpix, correct_pipeline_without_refinement, run_pipeline, tmp_path, request
):
    """
    Test that validity_mask is correctly saved when pandora2d pipeline is executed
    """

    input_cfg = request.getfixturevalue(input_cfg)
    correct_pipeline_without_refinement["pipeline"]["matching_cost"]["step"] = step
    correct_pipeline_without_refinement["pipeline"]["matching_cost"]["subpix"] = subpix

    configuration = {**input_cfg, **correct_pipeline_without_refinement, **{"output": {"path": str(tmp_path)}}}

    run_pipeline(configuration)

    validity_mask_path = tmp_path / "disparity_map" / "validity.tif"

    # band names correspond to criteria names (except for the first, which corresponds to valid points
    # and the last one which corresponds to P2D_DISPARITY_UNPROCESSED.)
    # and to the global “validity_mask” band
    expected_band_names = tuple(["validity_mask"] + list(Criteria.__members__.keys())[1:-1])

    # Check that validity_mask.tif exists
    assert (validity_mask_path).exists()

    # Check that validity_mask.tif contains nine bands of type uint8 with correct names
    with rasterio.open(validity_mask_path) as dataset:
        assert dataset.count == 8
        assert all(dtype == "uint8" for dtype in dataset.dtypes)
        assert dataset.descriptions == expected_band_names


@pytest.mark.parametrize(
    ["input_cfg", "step", "subpix"],
    [
        pytest.param(
            "correct_input_cfg",
            [1, 1],
            1,
            id="No mask and step=[1,1]",
        ),
        pytest.param(
            "correct_input_with_right_mask",
            [1, 3],
            1,
            id="Right mask and step=[1,3]",
        ),
        pytest.param(
            "correct_input_with_left_right_mask",
            [4, 5],
            2,
            id="Left and right masks, step=[4,5] and subpix=2",
        ),
        pytest.param(
            "correct_input_with_left_mask",
            [2, 1],
            4,
            id="Left mask, step=[2,1] and subpix=4",
        ),
    ],
)
def test_validity_mask_saved_with_roi(
    input_cfg, step, subpix, correct_pipeline_without_refinement, run_pipeline, tmp_path, request
):
    """
    Test that validity_mask is correctly saved when pandora2d pipeline is executed with a ROI
    """

    input_cfg = request.getfixturevalue(input_cfg)
    correct_pipeline_without_refinement["pipeline"]["matching_cost"]["step"] = step
    correct_pipeline_without_refinement["pipeline"]["matching_cost"]["subpix"] = subpix

    configuration = {
        **input_cfg,
        "ROI": {"col": {"first": 10, "last": 100}, "row": {"first": 10, "last": 100}},
        **correct_pipeline_without_refinement,
        **{"output": {"path": str(tmp_path)}},
    }

    run_pipeline(configuration)

    validity_mask_path = tmp_path / "disparity_map" / "validity.tif"

    # band names correspond to criteria names (except for the first, which corresponds to valid points
    # and the last one which corresponds to P2D_DISPARITY_UNPROCESSED.)
    # and to the global “validity_mask” band
    expected_band_names = tuple(["validity_mask"] + list(Criteria.__members__.keys())[1:-1])

    # Check that validity_mask.tif exists
    assert (validity_mask_path).exists()

    # Check that validity_mask.tif contains eight bands of type uint8 with correct names
    with rasterio.open(validity_mask_path) as dataset:
        assert dataset.count == 8
        assert all(dtype == "uint8" for dtype in dataset.dtypes)
        assert dataset.descriptions == expected_band_names
