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

# pylint: disable=redefined-outer-name

import pytest

# Mark all test of the module with monitor_test
pytestmark = pytest.mark.monitor_test


@pytest.fixture(params=[1, 20, 50, 100])
def sample_factor(request):
    return request.param


@pytest.fixture(params=[3, 5, 9, 11])
def range_col(request):
    return request.param


@pytest.fixture(params=[3, 5, 9, 11])
def range_row(request):
    return request.param


@pytest.mark.estimation_resource_tests
@pytest.mark.parametrize(
    ["left_img", "right_img"],
    [
        pytest.param("small_left_img_path", "small_right_img_path", id="image.size=(224, 186)"),
        pytest.param("large_left_img_path", "large_right_img_path", id="image.size=(2000, 2000)"),
    ],
    indirect=True,
)
def test_estimation(run_pipeline, input_cfg_for_estimation, tmp_path, sample_factor, range_col, range_row):
    """Test a configuration with only an estimation in the pipeline."""
    configuration = {
        **input_cfg_for_estimation,
        "pipeline": {
            "estimation": {
                "estimation_method": "phase_cross_correlation",
                "range_col": range_col,
                "range_row": range_row,
                "sample_factor": sample_factor,
            },
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)
