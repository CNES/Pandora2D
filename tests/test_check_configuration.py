#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Test configuration
"""
import unittest
from typing import Any, Dict

import pytest
import transitions
from json_checker import DictCheckerError

from pandora2d import check_configuration
from pandora2d.state_machine import Pandora2DMachine
from tests import common


class TestCheckJson(unittest.TestCase):
    """
    TestCheckJson class allows to test all the methods in the class CheckJson
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """

    @staticmethod
    def test_check_input_section() -> None:
        """
        Test function for checking user input section
        """

        assert check_configuration.check_input_section(common.correct_input)

        with pytest.raises(SystemExit):
            check_configuration.check_input_section(common.false_input_disp)
        with pytest.raises(DictCheckerError):
            check_configuration.check_input_section(common.false_input_path_image)

    @staticmethod
    def test_check_pipeline_section() -> None:
        """
        Test function for checking user pipeline section
        """

        pandora2d_machine = Pandora2DMachine()

        assert check_configuration.check_pipeline_section(common.correct_pipeline_dict, pandora2d_machine)

        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(common.false_pipeline_mc_dict, pandora2d_machine)
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(common.false_pipeline_disp_dict, pandora2d_machine)

    @staticmethod
    def test_check_roi_section() -> None:
        """
        Test function for checking user ROI section
        """

        # with a correct ROI check_roi_section should return nothing
        assert check_configuration.check_roi_section(common.correct_ROI_sensor)

        # if a dimension < 0 check_roi_section should raise BaseException error
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(common.false_ROI_sensor_negative)
        # if a dimension first > last check_roi_section should raise BaseException error
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(common.false_ROI_sensor_first_superior_to_last)

    @staticmethod
    def test_get_roi_pipeline() -> None:
        """
        Test get_roi_pipeline function
        """

        assert common.correct_ROI_sensor == check_configuration.get_roi_config(common.correct_ROI_sensor)

    @staticmethod
    def test_check_roi_coherence() -> None:
        """
        Test check_roi_coherence function
        """

        # if first < last check_roi_coherence should return None
        assert check_configuration.check_roi_coherence(common.correct_ROI_sensor["ROI"]["col"]) is None

        # if first > last check_roi_coherence should raise SystemExit error
        with pytest.raises(SystemExit):
            check_configuration.check_roi_coherence(common.false_ROI_sensor_first_superior_to_last["ROI"]["col"])

    def test_check_step(self) -> None:
        """
        Test step configuration with user configuration dictionary
        """
        pandora2d_machine = Pandora2DMachine()

        pipeline_cfg: Dict[str, Dict[str, Any]] = common.correct_pipeline_dict

        # Add correct step
        pipeline_cfg["pipeline"]["matching_cost"]["step"] = [1, 1]
        assert check_configuration.check_pipeline_section(pipeline_cfg, pandora2d_machine)

        pandora2d_machine = Pandora2DMachine()

        # Test with a one size list step : test should fail
        pipeline_cfg["pipeline"]["matching_cost"]["step"] = [1]
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_cfg, pandora2d_machine)

        pandora2d_machine = Pandora2DMachine()

        # Test with a negative step : test should fail
        pipeline_cfg["pipeline"]["matching_cost"]["step"] = [-1, 1]
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_cfg, pandora2d_machine)

        pandora2d_machine = Pandora2DMachine()

        # Test with a three elements list step : test should fail
        pipeline_cfg["pipeline"]["matching_cost"]["step"] = [1, 1, 1]
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_cfg, pandora2d_machine)

        pandora2d_machine = Pandora2DMachine()

        # Test with a str elements list step : test should fail
        pipeline_cfg["pipeline"]["matching_cost"]["step"] = [1, "1"]
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_cfg, pandora2d_machine)
