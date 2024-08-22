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
#

"""
Test used resources during execution of a configuration.
"""

import pytest


# Mark all test of the module with monitor_test
pytestmark = pytest.mark.monitor_test


def test_estimation(run_pipeline, correct_input_cfg):
    """Test a configuration with only an estimation in the pipeline."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "estimation": {"estimation_method": "phase_cross_correlation"},
        },
    }
    run_pipeline(configuration)


@pytest.mark.parametrize("subpix", [1, 2, 4])
@pytest.mark.parametrize("matching_cost_method", ["zncc", "sad", "ssd"])
def test_matching_cost_with_disparity(run_pipeline, correct_input_cfg, matching_cost_method, subpix):
    """Test pipeline with a matching_cost and a disparity steps."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
    }
    run_pipeline(configuration)


@pytest.mark.parametrize("subpix", [1, 2, 4])
@pytest.mark.parametrize("matching_cost_method", ["zncc", "sad", "ssd"])
def test_matching_cost_with_estimation_and_disparity(run_pipeline, correct_input_cfg, matching_cost_method, subpix):
    """Test pipeline with an estimation, a matching_cost and a disparity steps."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "estimation": {"estimation_method": "phase_cross_correlation"},
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
    }
    run_pipeline(configuration)
