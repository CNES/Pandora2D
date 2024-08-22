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
