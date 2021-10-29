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
import rasterio

from pandora2d import check_json
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

        assert check_json.check_input_section(common.correct_input)

        with pytest.raises(rasterio.errors.RasterioIOError):
            check_json.check_input_section(common.false_input_path_image)

        with pytest.raises(BaseException):
            check_json.check_input_section(common.false_input_no_data)
            check_json.check_input_section(common.false_input_disp)
