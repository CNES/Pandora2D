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
Test state_machine
"""
import unittest
import copy

import pytest


from transitions.core import MachineError
from pandora import read_img

from tests import common
from pandora2d import state_machine


class TestPandora2D(unittest.TestCase):
    """
    TestCheckJson class allows to test all the methods in the class CheckJson
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """

    @staticmethod
    def test_run_pandora() -> None:
        """
        Test function for checking user input section
        """

        pandora2d_machine = state_machine.Pandora2DMachine()

        correct_cfg = copy.deepcopy(common.correct_pipeline)
        pandora2d_machine.check_conf(correct_cfg) # type: ignore

        false_cfg_mc = copy.deepcopy(common.false_pipeline_mc)
        false_cfg_disp = copy.deepcopy(common.false_pipeline_disp)
        with pytest.raises(MachineError):
            pandora2d_machine.check_conf(false_cfg_mc) # type: ignore
            pandora2d_machine.check_conf(false_cfg_disp) # type: ignore

    @staticmethod
    def test_run_prepare() -> None:
        """
        Test run_prepare function
        """
        pandora2d_machine = state_machine.Pandora2DMachine()

        img_left = read_img("./tests/data/left.png", -9999)
        img_right = read_img("./tests/data/right.png", -9999)

        pandora2d_machine.run_prepare(img_left, img_right, -2, 2, -2, 2)

        assert pandora2d_machine.left_img == img_left
        assert pandora2d_machine.right_img == img_right
        assert pandora2d_machine.disp_min_row == -2
        assert pandora2d_machine.disp_max_row == 2
        assert pandora2d_machine.disp_min_col == -2
        assert pandora2d_machine.disp_max_col == 2


