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
import pytest

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

        with pytest.raises(BaseException):
            check_configuration.check_input_section(common.false_input_disp)
        with pytest.raises(BaseException):
            check_configuration.check_input_section(common.false_input_path_image)

    @staticmethod
    def test_check_pipeline_section() -> None:
        """
        Test function for checking user pipeline section
        """

        pandora2d_machine = Pandora2DMachine()

        assert check_configuration.check_pipeline_section(common.correct_pipeline_dict, pandora2d_machine)

        with pytest.raises(BaseException):
            check_configuration.check_pipeline_section(common.false_pipeline_mc_dict, pandora2d_machine)
        with pytest.raises(BaseException):
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

