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
Test used resources during execution of a configuration.
"""

import pytest

# Mark all test of the module with monitor_test
pytestmark = pytest.mark.monitor_test


@pytest.mark.matching_cost_resource_tests
@pytest.mark.parametrize(
    ["left_img", "right_img", "step", "window_size"],
    [
        pytest.param("small_left_img_path", "small_right_img_path", [1, 1], 5, id="image.size=(224, 186)"),
        pytest.param("large_left_img_path", "large_right_img_path", [16, 16], 33, id="image.size=(2000, 2000)"),
    ],
    indirect=True,
)
def test_matching_cost_with_disparity(
    run_pipeline, input_cfg, matching_cost_method, window_size, step, subpix, tmp_path
):
    """Test pipeline with a matching_cost and a disparity steps."""
    configuration = {
        **input_cfg,
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": matching_cost_method,
                "window_size": window_size,
                "step": step,
                "subpix": subpix,
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)


@pytest.mark.matching_cost_resource_tests
@pytest.mark.parametrize(
    ["left_img", "right_img", "step"],
    [
        pytest.param("small_left_img_path", "small_right_img_path", [1, 1], id="image.size=(224, 186)"),
        pytest.param("large_left_img_path", "large_right_img_path", [16, 16], id="image.size=(2000, 2000)"),
    ],
    indirect=True,
)
@pytest.mark.matching_cost_resource_tests
def test_matching_cost_with_estimation_and_disparity(
    run_pipeline, input_cfg_for_estimation, matching_cost_method, subpix, step, tmp_path
):
    """Test pipeline with an estimation, a matching_cost and a disparity steps."""
    configuration = {
        **input_cfg_for_estimation,
        "pipeline": {
            "estimation": {"estimation_method": "phase_cross_correlation"},
            "matching_cost": {
                "matching_cost_method": matching_cost_method,
                "window_size": 33,
                "step": step,
                "subpix": subpix,
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)
