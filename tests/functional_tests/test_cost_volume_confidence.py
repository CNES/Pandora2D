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
Functional tests for configurations with cost_volume_confidence step.
"""

# pylint: disable=redefined-outer-name

import copy
import sys
import numpy as np
import pytest
import rasterio

import pandora2d
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d.state_machine import Pandora2DMachine


@pytest.fixture()
def make_cfg_for_confidence(
    correct_input_for_functional_tests,
    cost_volume_confidence_method,
    eta_max,
    eta_step,
    normalization,
    step,
    subpix,
):
    """
    Creates user configuration to test dichotomy loop
    """

    user_cfg = {
        **correct_input_for_functional_tests,
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": 7,
                "subpix": subpix,
                "step": step,
            },
            "cost_volume_confidence": {
                "confidence_method": cost_volume_confidence_method,
                "eta_max": eta_max,
                "eta_step": eta_step,
                "normalization": normalization,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
        },
        "output": {"path": "home"},
    }

    return user_cfg


@pytest.mark.parametrize("cost_volume_confidence_method", ["ambiguity"])
class TestAmbiguity:
    """
    Test end-to-end tests for ambiguity with different configurations
    """

    @pytest.fixture
    def eta_max(self):
        return 0.7

    @pytest.fixture
    def eta_step(self):
        return 0.01

    @pytest.fixture
    def normalization(self):
        return True

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Does not work on windows")
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
    @pytest.mark.parametrize("eta_max", [0.7, 0.8, 0.9, 0.99])
    @pytest.mark.parametrize("eta_step", [0.01, 0.1, 0.2, 0.5])
    @pytest.mark.parametrize("normalization", [True, False])
    def test_run(self, make_cfg_for_confidence):
        """
        Docstring : Test pipeline with ambiguity

        Data :
        * Left_img : cones/monoband/left.png
        * Right_img : cones/monoband/right.png
        """

        pandora2d_machine = Pandora2DMachine()

        user_cfg = copy.deepcopy(make_cfg_for_confidence)
        cfg = check_conf(user_cfg, pandora2d_machine)

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"])

        dataset_disp_maps, _ = pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

        confidence_data = dataset_disp_maps.confidence_measure.data

        # Checking that resulting confidence_measure map are not full of 0 or same value (confidence_data.flat[0])
        assert not np.all(confidence_data == 0)
        assert not np.all(confidence_data == confidence_data.flat[0])


class TestCostVolumeConfidence:
    """
    Test cost volume confidence execution
    """

    @pytest.fixture()
    def pipeline_cfg_with_confidence(self):
        """
        Cost volume confidence configuration
        """

        return {
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "zncc",
                    "window_size": 5,
                },
                "cost_volume_confidence": {"confidence_method": "ambiguity", "eta_max": 0.7},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            }
        }

    @pytest.fixture()
    def configuration(self, correct_input_cfg, pipeline_cfg_with_confidence, tmp_path):
        return {
            **correct_input_cfg,
            **pipeline_cfg_with_confidence,
            **{"output": {"path": str(tmp_path)}},
        }

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Does not work on windows")
    def test_cost_volume_confidence_pipeline(self, configuration, run_pipeline, tmp_path):
        """
        Test execution of a pipeline with cost volumes confidence
        """

        run_pipeline(configuration)

        with rasterio.open(tmp_path / "cost_volumes" / "confidence_measure.tif") as src:
            confidence_map = src.read(1)

        # Checking that resulting confidence map is not full of nans
        assert not np.all(np.isnan(confidence_map))
