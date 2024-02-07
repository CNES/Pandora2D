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
import copy
import numpy as np

import pytest


from transitions.core import MachineError

from pandora.margins import Margins
from tests import common
from pandora2d import state_machine
from pandora2d.img_tools import create_datasets_from_inputs


class TestPandora2D:
    """
    TestCheckJson class allows to test all the methods in the class CheckJson
    """

    @staticmethod
    def test_run_pandora() -> None:
        """
        Test function for checking user input section
        """

        pandora2d_machine = state_machine.Pandora2DMachine()

        correct_cfg = copy.deepcopy(common.correct_pipeline)
        pandora2d_machine.check_conf(correct_cfg)

        false_cfg_mc = copy.deepcopy(common.false_pipeline_mc)
        false_cfg_disp = copy.deepcopy(common.false_pipeline_disp)
        with pytest.raises(MachineError):
            pandora2d_machine.check_conf(false_cfg_mc)
            pandora2d_machine.check_conf(false_cfg_disp)

    @staticmethod
    def test_run_prepare() -> None:
        """
        Test run_prepare function
        """
        pandora2d_machine = state_machine.Pandora2DMachine()

        input_config = {
            "left": {"img": "./tests/data/left.png", "nodata": -9999},
            "right": {"img": "./tests/data/right.png", "nodata": -9999},
            "col_disparity": [-2, 2],
            "row_disparity": [-2, 2],
        }
        img_left, img_right = create_datasets_from_inputs(input_config=input_config)

        pandora2d_machine.run_prepare(img_left, img_right)

        assert pandora2d_machine.left_img == img_left
        assert pandora2d_machine.right_img == img_right
        np.testing.assert_array_equal(
            pandora2d_machine.disp_min_col, np.full((img_left.sizes["row"], img_left.sizes["col"]), -2)
        )
        np.testing.assert_array_equal(
            pandora2d_machine.disp_max_col, np.full((img_left.sizes["row"], img_left.sizes["col"]), 2)
        )
        np.testing.assert_array_equal(
            pandora2d_machine.disp_min_row, np.full((img_left.sizes["row"], img_left.sizes["col"]), -2)
        )
        np.testing.assert_array_equal(
            pandora2d_machine.disp_max_row, np.full((img_left.sizes["row"], img_left.sizes["col"]), 2)
        )

    @staticmethod
    def test_global_margins() -> None:
        """
        Test computed global margins is as expected.
        """

        pandora2d_machine = state_machine.Pandora2DMachine()

        pipeline_cfg = copy.deepcopy(common.correct_pipeline)
        pandora2d_machine.check_conf(pipeline_cfg)

        assert pandora2d_machine.margins.global_margins == Margins(3, 3, 3, 3)
