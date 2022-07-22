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
Test refinement step
"""

import unittest
import pytest
import json_checker
from pandora2d import estimation


class TestEstimation(unittest.TestCase):
    """
    TestEstimation class allows to test the estimation module
    """

    @staticmethod
    def test_check_conf():
        """
        Test the estimation methods
        """

        estimation.AbstractEstimation(**{"estimation_method": "phase_cross_correlation"})  # type: ignore

        with pytest.raises(KeyError):
            estimation.AbstractEstimation(**{"estimation_method": "wta"})  # type: ignore

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            estimation.AbstractEstimation(  # type: ignore
                **{"estimation_method": "phase_cross_correlation", "range_col": 5, "range_row": "10"}  # type: ignore
            )

    @staticmethod
    def test_ceil_or_floor():
        """
        test ceil_or_floor method
        """

        estim_ = estimation.AbstractEstimation(**{"estimation_method": "phase_cross_correlation"})  # type: ignore

        assert estim_.ceil_or_floor(2.15) == 3
        assert estim_.ceil_or_floor(-2.15) == -3
