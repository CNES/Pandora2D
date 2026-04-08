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
#

"""
Test used resources during execution of a configuration.
"""

# pylint: disable=redefined-outer-name

import pytest

# Mark all test of the module with monitor_test
pytestmark = pytest.mark.monitor_test


@pytest.fixture
def dichotomy_pipeline(
    matching_cost_method,
    window_size,
    step,
    subpix,
    iterations,
    dicho_method,
    filter_method,
):
    """Pipeline for a dichotomy refinement."""
    return {
        "matching_cost": {
            "matching_cost_method": matching_cost_method,
            "window_size": window_size,
            "step": step,
            "subpix": subpix,
        },
        "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        "refinement": {
            "refinement_method": dicho_method,
            "iterations": iterations,
            "filter": {"method": filter_method},
        },
    }


@pytest.mark.dichotomy_resource_tests
@pytest.mark.parametrize(
    ("dicho_method", "filter_method"),
    [
        ("dichotomy", "bicubic"),
        ("dichotomy", "sinc"),
    ],
)
@pytest.mark.parametrize(
    "matching_cost_method",
    [
        pytest.param(
            "mutual_information",
            marks=pytest.mark.skip(reason="This method takes too long for these tests. See ticket 371."),
        ),
        "zncc",
    ],
)
@pytest.mark.parametrize(
    ["left_img", "right_img", "step", "window_size"],
    [
        pytest.param("small_left_img_path", "small_right_img_path", [1, 1], 5, id="image.size=(224, 186)"),
    ],
    indirect=True,
)
def test_dichotomy_with_small_img(run_pipeline, input_cfg, dichotomy_pipeline, tmp_path):
    """Test dichotomy with an image size of 224x186"""
    configuration = {
        **input_cfg,
        "pipeline": {
            **dichotomy_pipeline,
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)


@pytest.mark.dichotomy_resource_tests
@pytest.mark.parametrize(
    ("dicho_method", "filter_method"),
    [
        ("dichotomy", "bicubic"),
        ("dichotomy", "sinc"),
    ],
)
@pytest.mark.parametrize(
    "matching_cost_method",
    [
        pytest.param(
            "mutual_information",
            marks=pytest.mark.skip(reason="This method takes too long for these tests. See ticket 371."),
        ),
        "zncc",
    ],
)
@pytest.mark.parametrize(
    ["left_img", "right_img", "step", "window_size"],
    [pytest.param("large_left_img_path", "large_right_img_path", [16, 16], 33, id="image.size=(2000, 2000)")],
    indirect=True,
)
@pytest.mark.parametrize(
    "subpix",
    [
        1,
        pytest.param(2, marks=pytest.mark.skip(reason="Memory capacity explodes with this subpix.")),
        pytest.param(4, marks=pytest.mark.skip(reason="Memory capacity explodes with this subpix.")),
    ],
)
def test_dichotomy_with_large_img(run_pipeline, input_cfg, dichotomy_pipeline, tmp_path):
    """Test dichotomy with an image size of 2000x2000"""
    configuration = {
        **input_cfg,
        "pipeline": {
            **dichotomy_pipeline,
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)
