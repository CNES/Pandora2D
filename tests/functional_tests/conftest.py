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
"""Module with global test fixtures."""

import json

import pytest

import pandora2d


@pytest.fixture()
def run_pipeline(tmp_path):
    """Fixture that returns a function to run a pipeline and which returns the output directory path."""

    def run(configuration, output_dir="output"):
        config_path = tmp_path / "config.json"
        with config_path.open("w", encoding="utf-8") as file_:
            json.dump(configuration, file_, indent=2)

        pandora2d.main(str(config_path), str(tmp_path / output_dir), verbose=False)
        return tmp_path

    return run


@pytest.fixture()
def correct_pipeline_without_refinement():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        }
    }


@pytest.fixture()
def correct_pipeline_with_optical_flow():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "optical_flow"},
        }
    }
