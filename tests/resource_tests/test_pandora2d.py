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

import numpy as np
import pytest
from PIL import Image

# Mark all test of the module with monitor_test
pytestmark = pytest.mark.monitor_test

subpix_list = [1, 2, 4]
matching_cost_methods = ["zncc", "sad", "ssd", "mutual_information"]
iteration_list = [1, 4, 9]


def reduce_image(input_path, output_path):
    data = np.asarray(Image.open(input_path))
    half_row, half_col = data.shape[0] // 2, data.shape[1] // 2
    image = Image.fromarray(data[half_row - 25 : half_row + 25, half_col - 25 : half_col + 25])
    image.save(output_path, "png")


@pytest.fixture()
def left_img_path(tmp_path, left_img_path):
    path = tmp_path / "left.png"
    reduce_image(left_img_path, path)
    return str(path)


@pytest.fixture()
def right_img_path(tmp_path, right_img_path):
    path = tmp_path / "right.png"
    reduce_image(right_img_path, path)
    return str(path)


def test_estimation(run_pipeline, correct_input_cfg, tmp_path):
    """Test a configuration with only an estimation in the pipeline."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "estimation": {"estimation_method": "phase_cross_correlation"},
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)


@pytest.mark.parametrize("subpix", subpix_list)
@pytest.mark.parametrize("matching_cost_method", matching_cost_methods)
def test_matching_cost_with_disparity(run_pipeline, correct_input_cfg, matching_cost_method, subpix, tmp_path):
    """Test pipeline with a matching_cost and a disparity steps."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)


@pytest.mark.parametrize("subpix", subpix_list)
@pytest.mark.parametrize("matching_cost_method", matching_cost_methods)
def test_matching_cost_with_estimation_and_disparity(
    run_pipeline, correct_input_cfg, matching_cost_method, subpix, tmp_path
):
    """Test pipeline with an estimation, a matching_cost and a disparity steps."""
    configuration = {
        **correct_input_cfg,
        "pipeline": {
            "estimation": {"estimation_method": "phase_cross_correlation"},
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        },
        "output": {"path": str(tmp_path)},
    }
    run_pipeline(configuration)


@pytest.mark.parametrize("subpix", subpix_list)
@pytest.mark.parametrize("matching_cost_method", matching_cost_methods)
class TestRefinement:
    """Test pipelines which include a refinement step."""

    @pytest.fixture()
    def dichotomy_pipeline(self, matching_cost_method, subpix, iterations, dicho_method, filter_method):
        """Pipeline for a dichotomy refinement."""
        return {
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {
                "refinement_method": dicho_method,
                "iterations": iterations,
                "filter": {"method": filter_method},
            },
        }

    @pytest.fixture()
    def optical_flow_pipeline(self, matching_cost_method, subpix, iterations):
        """Pipeline for an optical flow refinement."""
        return {
            "matching_cost": {"matching_cost_method": matching_cost_method, "subpix": subpix},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {
                "refinement_method": "optical_flow",
                "iterations": iterations,
            },
        }

    @pytest.mark.parametrize("iterations", iteration_list)
    @pytest.mark.parametrize(
        ("dicho_method", "filter_method"),
        [
            ("dichotomy_python", "bicubic_python"),
            ("dichotomy_python", "sinc_python"),
            ("dichotomy", "bicubic"),
            ("dichotomy", "sinc"),
        ],
    )
    def test_dichotomy(self, run_pipeline, correct_input_cfg, dichotomy_pipeline, tmp_path):
        """Test dichotomy."""
        configuration = {
            **correct_input_cfg,
            "pipeline": {
                **dichotomy_pipeline,
            },
            "output": {"path": str(tmp_path)},
        }
        run_pipeline(configuration)

    @pytest.mark.parametrize("iterations", iteration_list)
    @pytest.mark.parametrize(
        ("dicho_method", "filter_method"),
        [
            ("dichotomy_python", "bicubic_python"),
            ("dichotomy_python", "sinc_python"),
            ("dichotomy", "bicubic"),
            ("dichotomy", "sinc"),
        ],
    )
    def test_dichotomy_with_estimation(self, run_pipeline, correct_input_cfg, dichotomy_pipeline, tmp_path):
        """Test dichotomy with estimation."""
        configuration = {
            **correct_input_cfg,
            "pipeline": {
                "estimation": {"estimation_method": "phase_cross_correlation"},
                **dichotomy_pipeline,
            },
            "output": {"path": str(tmp_path)},
        }
        run_pipeline(configuration)

    @pytest.mark.parametrize("iterations", iteration_list)
    def test_optical_flows(self, run_pipeline, correct_input_cfg, optical_flow_pipeline, tmp_path):
        """Test optical flows."""
        configuration = {
            **correct_input_cfg,
            "pipeline": {
                **optical_flow_pipeline,
            },
            "output": {"path": str(tmp_path)},
        }
        run_pipeline(configuration)

    @pytest.mark.parametrize("iterations", iteration_list)
    def test_optical_flows_with_estimation(self, run_pipeline, correct_input_cfg, optical_flow_pipeline, tmp_path):
        """Test optical flows with estimation."""
        configuration = {
            **correct_input_cfg,
            "pipeline": {
                "estimation": {"estimation_method": "phase_cross_correlation"},
                **optical_flow_pipeline,
            },
            "output": {"path": str(tmp_path)},
        }
        run_pipeline(configuration)
