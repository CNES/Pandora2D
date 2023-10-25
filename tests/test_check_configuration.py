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

import pytest
import transitions
from json_checker import DictCheckerError

from pandora2d import check_configuration
from tests import common


class TestCheckInputSection:
    """
    Test check input section.
    """

    def test_check_nominal_case(self) -> None:
        assert check_configuration.check_input_section(common.correct_input)

    def test_false_input_disp_should_exit(self):
        with pytest.raises(SystemExit):
            check_configuration.check_input_section(common.false_input_disp)

    def test_false_input_path_image_should_raise_error(self):
        with pytest.raises(DictCheckerError):
            check_configuration.check_input_section(common.false_input_path_image)


class TestCheckPipelineSection:
    """Test check_pipeline_section."""

    def test_nominal_case(self, pandora2d_machine) -> None:
        """
        Test function for checking user pipeline section
        """
        assert check_configuration.check_pipeline_section(common.correct_pipeline, pandora2d_machine)

    def test_false_mc_dict_should_raise_error(self, pandora2d_machine):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(common.false_pipeline_mc, pandora2d_machine)

    def test_false_disp_dict_should_raise_error(self, pandora2d_machine):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(common.false_pipeline_disp, pandora2d_machine)


class TestCheckRoiSection:
    """Test check_roi_section."""

    def test_nominal_case(self) -> None:
        """
        Test function for checking user ROI section
        """
        # with a correct ROI check_roi_section should return nothing
        assert check_configuration.check_roi_section(common.correct_ROI_sensor)

    def test_dimension_lt_0_raises_exception(self):
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(common.false_ROI_sensor_negative)

    def test_first_dimension_gt_last_dimension_raises_exception(self):
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(common.false_ROI_sensor_first_superior_to_last)


def test_get_roi_pipeline() -> None:
    """
    Test get_roi_pipeline function
    """
    assert common.correct_ROI_sensor == check_configuration.get_roi_config(common.correct_ROI_sensor)


class TestCheckRoiCoherence:
    """Test check_roi_coherence."""

    def test_first_lt_last_is_ok(self) -> None:
        check_configuration.check_roi_coherence(common.correct_ROI_sensor["ROI"]["col"])

    def test_first_gt_last_raises_error(self):
        with pytest.raises(SystemExit):
            check_configuration.check_roi_coherence(common.false_ROI_sensor_first_superior_to_last["ROI"]["col"])


class TestCheckStep:
    """Test check_step."""

    def test_nominal_case(self, pipeline_config, pandora2d_machine) -> None:
        """
        Test step configuration with user configuration dictionary
        """

        # Add correct step
        pipeline_config["pipeline"]["matching_cost"]["step"] = [1, 1]
        assert check_configuration.check_pipeline_section(pipeline_config, pandora2d_machine)

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param([1], id="one size list"),
            pytest.param([-1, 1], id="negative value"),
            pytest.param([1, 1, 1], id="More than 2 elements"),
            pytest.param([1, "1"], id="String element"),
        ],
    )
    def test_fails_with_bad_step_values(self, pipeline_config, pandora2d_machine, step) -> None:
        """Test check_pipeline_section fails with bad values of step."""
        pipeline_config["pipeline"]["matching_cost"]["step"] = step
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_config, pandora2d_machine)


class TestCheckConfMatchingCostNodataCondition:
    """Test check conf for right image’s nodata."""

    @pytest.fixture()
    def build_configuration(self):
        """Return a builder for configuration."""

        def function(right_nodata, matching_cost_method):
            return {
                "input": {
                    "left": {
                        "img": "./tests/data/left.png",
                        "nodata": "NaN",
                    },
                    "right": {
                        "img": "./tests/data/right.png",
                        "nodata": right_nodata,
                    },
                    "col_disparity": [-2, 2],
                    "row_disparity": [-2, 2],
                },
                "pipeline": {
                    "matching_cost": {"matching_cost_method": matching_cost_method, "window_size": 1},
                },
            }

        return function

    @pytest.mark.parametrize("right_nodata", ["NaN", 0.1, "inf", None])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd"])
    def test_sad_or_ssd_fail_with(self, pandora2d_machine, build_configuration, matching_cost_method, right_nodata):
        """Right nodata must be an integer with sad or ssd matching_cost_method."""
        configuration = build_configuration(right_nodata, matching_cost_method)
        with pytest.raises((SystemExit, DictCheckerError)):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc"])
    def test_passes_with_int(self, pandora2d_machine, build_configuration, matching_cost_method):
        """Right nodata must be an integer."""
        configuration = build_configuration(432, matching_cost_method)
        check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("right_nodata", ["NaN", "inf"])
    def test_zncc_passes_with(self, pandora2d_machine, build_configuration, right_nodata):
        """Right nodata can be inf or nan with zncc matching_cost_method."""
        configuration = build_configuration(right_nodata, "zncc")
        check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("right_nodata", [0.2, None])
    def test_zncc_fails_with(self, pandora2d_machine, build_configuration, right_nodata):
        """Right nodata must can not be float or nan with zncc matching_cost_method."""
        configuration = build_configuration(right_nodata, "zncc")
        with pytest.raises((SystemExit, DictCheckerError)):
            check_configuration.check_conf(configuration, pandora2d_machine)
