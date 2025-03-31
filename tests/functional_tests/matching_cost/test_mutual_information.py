#!/usr/bin/env python
#
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
Tests pandora2d machine execution with mutual information
"""

import pytest
import numpy as np

import pandora2d
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing


class TestMutualInformation:
    """
    Test that the pandora2d machine runs correctly with the mutual information method
    for different parameter panels
    """

    @pytest.fixture()
    def make_cfg_for_mutual_information(
        self,
        correct_input_for_functional_tests,
        window_size,
        subpix,
        step,
        roi,
    ):
        """
        Creates user configuration to test dichotomy loop
        """

        user_cfg = {
            **correct_input_for_functional_tests,
            "ROI": roi,
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "mutual_information",
                    "window_size": window_size,
                    "subpix": subpix,
                    "step": step,
                },
                "disparity": {
                    "disparity_method": "wta",
                    "invalid_disparity": -9999,
                },
            },
            "output": {"path": "where"},
        }

        return user_cfg

    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize("window_size", [1, 3, 5])
    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
    @pytest.mark.parametrize("roi", [{"col": {"first": 100, "last": 120}, "row": {"first": 100, "last": 120}}])
    @pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 1}])
    @pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 3}])
    def test_mutual_information_execution(self, make_cfg_for_mutual_information):
        """
        Description : Test that execution of Pandora2d with mutual information does not fail.
        Data :
            * Left_img : cones/monoband/left.png
            * Right_img : cones/monoband/right.png
        """
        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(make_cfg_for_mutual_information, pandora2d_machine)

        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

        dataset_disp_maps, _ = pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

        # Checking that resulting disparity maps are not full of nans
        with np.testing.assert_raises(AssertionError):
            assert np.all(np.isnan(dataset_disp_maps.row_map.data))
            assert np.all(np.isnan(dataset_disp_maps.col_map.data))
