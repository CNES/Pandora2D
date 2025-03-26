#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
Test state_machine
"""

import copy
import pytest

from transitions.core import MachineError

from pandora.margins import Margins
from pandora2d import state_machine
from pandora2d.img_tools import create_datasets_from_inputs


class TestPandora2D:
    """
    TestCheckJson class allows to test all the methods in the class CheckJson
    """

    @staticmethod
    def test_run_pandora(correct_pipeline, false_pipeline_mc, false_pipeline_disp) -> None:
        """
        Description : Test function for checking user input section
        Data :
        Requirement : EX_CONF_08
        """

        pandora2d_machine = state_machine.Pandora2DMachine()

        correct_cfg = copy.deepcopy(correct_pipeline)
        pandora2d_machine.check_conf(correct_cfg)

        false_cfg_mc = copy.deepcopy(false_pipeline_mc)
        false_cfg_disp = copy.deepcopy(false_pipeline_disp)
        with pytest.raises(MachineError):
            pandora2d_machine.check_conf(false_cfg_mc)
            pandora2d_machine.check_conf(false_cfg_disp)

    @staticmethod
    def test_run_prepare(left_img_path, right_img_path) -> None:
        """
        Test run_prepare function
        """
        pandora2d_machine = state_machine.Pandora2DMachine()

        input_config = {
            "left": {"img": left_img_path, "nodata": -9999},
            "right": {"img": right_img_path, "nodata": -9999},
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 2},
        }
        img_left, img_right = create_datasets_from_inputs(input_config=input_config)

        pandora2d_machine.run_prepare(img_left, img_right, input_config)

        assert pandora2d_machine.left_img == img_left
        assert pandora2d_machine.right_img == img_right
        assert pandora2d_machine.completed_cfg == input_config

    @pytest.mark.parametrize(
        ["refinement_config", "expected"],
        [
            pytest.param(
                {"refinement_method": "dichotomy_python", "iterations": 3, "filter": {"method": "bicubic"}},
                Margins(1, 1, 2, 2),
                id="dichotomy python with bicubic filter",
            ),
            pytest.param(
                {"refinement_method": "dichotomy", "iterations": 3, "filter": {"method": "bicubic"}},
                Margins(1, 1, 2, 2),
                id="dichotomy cpp with bicubic filter",
            ),
        ],
    )
    def test_global_margins_disp(self, refinement_config, expected) -> None:
        """
        Test computed global margins for cost volume as expected.
        """

        pipeline_cfg = {
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
                "refinement": refinement_config,
            },
        }

        pandora2d_machine = state_machine.Pandora2DMachine()

        pandora2d_machine.check_conf(pipeline_cfg)

        assert pandora2d_machine.margins_disp.global_margins == expected

    @pytest.mark.parametrize(
        ["matching_cost_config", "expected"],
        [
            pytest.param({"matching_cost_method": "zncc"}, Margins(2, 2, 2, 2), id="zncc"),
        ],
    )
    def test_global_margins_img(self, matching_cost_config, expected) -> None:
        """
        Test computed global margins for image as expected.
        """

        pipeline_cfg = {
            "pipeline": {
                "matching_cost": matching_cost_config,
                "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            },
        }

        pandora2d_machine = state_machine.Pandora2DMachine()

        pandora2d_machine.check_conf(pipeline_cfg)

        assert pandora2d_machine.margins_img.global_margins == expected
