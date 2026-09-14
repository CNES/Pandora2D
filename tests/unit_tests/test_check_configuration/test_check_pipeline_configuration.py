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
#

"""
Test check pipeline step configuration
"""

import importlib.util
import pytest
import transitions
from json_checker import DictCheckerError

from pandora2d.check_configuration import check_pipeline_section

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name,unused-argument


class TestCheckPipelineSection:
    """Test check_pipeline_section."""

    def test_fails_if_pipeline_section_is_missing(self, pandora2d_machine) -> None:
        """
        Description : Test if the pipeline section is missing in the configuration file
        Data :
        Requirement : EX_CONF_02
        """
        with pytest.raises(KeyError, match="pipeline key is missing"):
            check_pipeline_section({}, pandora2d_machine)

    def test_nominal_case(self, pandora2d_machine, correct_pipeline) -> None:
        """
        Description : Test function for checking user pipeline section
        Data :
        Requirement : EX_REF_00
        """
        check_pipeline_section(correct_pipeline, pandora2d_machine)

    def test_false_mc_dict_should_raise_error(self, pandora2d_machine, false_pipeline_mc):
        """Test raises an error if the matching_cost key is missing"""
        with pytest.raises(transitions.core.MachineError):
            check_pipeline_section(false_pipeline_mc, pandora2d_machine)

    def test_false_disp_dict_should_raise_error(self, pandora2d_machine, false_pipeline_disp):
        """Test raises an error if the disparity key is missing"""
        with pytest.raises(transitions.core.MachineError):
            check_pipeline_section(false_pipeline_disp, pandora2d_machine)

    @pytest.mark.parametrize(
        "step_order",
        [
            ["disparity", "matching_cost", "refinement"],
            ["matching_cost", "refinement", "disparity"],
            ["matching_cost", "estimation", "disparity"],
            ["matching_cost", "disparity", "cost_volume_confidence"],
            ["cost_volume_confidence", "matching_cost", "disparity"],
        ],
    )
    def test_wrong_order_should_raise_error(self, pandora2d_machine, step_order):
        """
        Description : Pipeline section order is important.
        Data :
        Requirement : EX_CONF_07
        """
        steps = {
            "estimation": {"estimated_shifts": [-0.5, 1.3], "error": [1.0], "phase_diff": [1.0]},
            "matching_cost": {"matching_cost_method": "zncc_python", "window_size": 5},
            "cost_volume_confidence": {"confidence_method": "ambiguity"},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "dichotomy", "filter": {"method": "bicubic"}, "iterations": 2},
        }
        configuration = {"pipeline": {step: steps[step] for step in step_order}}
        with pytest.raises(transitions.core.MachineError):
            check_pipeline_section(configuration, pandora2d_machine)

    def test_multiband_pipeline(self, pandora2d_machine, left_rgb_path, right_rgb_path):
        """
        Description : Test the method check_pipeline_section for multiband images
        Data :
        - Left image : cones/multibands/left.tif
        - Right image : cones/multibands/right.tif
        Requirement : EX_CONF_12
        """
        input_multiband_cfg = {
            "left": {
                "img": left_rgb_path,
            },
            "right": {
                "img": right_rgb_path,
            },
            "col_disparity": {"init": -30, "range": 30},
            "row_disparity": {"init": -30, "range": 30},
        }
        cfg = {
            "input": input_multiband_cfg,
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc_python", "window_size": 5, "subpix": 2},
                "disparity": {"disparity_method": "wta"},
            },
            "output": {"path": "here"},
        }

        check_pipeline_section(cfg, pandora2d_machine)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ["pipeline_cfg"],
        [
            pytest.param("correct_pipeline_with_dichotomy_cpp", id="Dichotomy cpp with subpix=1"),
            pytest.param("correct_pipeline_with_dichotomy_python", id="Dichotomy python with subpix=1"),
        ],
    )
    def test_check_subpix_value_with_dichotomy(self, pipeline_cfg, pandora2d_machine, caplog, request):
        """
        Check a warning is raised when using dichotomy with a subpix equal to 1.
        """

        check_pipeline_section(request.getfixturevalue(pipeline_cfg), pandora2d_machine)

        assert (
            "To avoid aliasing, it is strongly recommended to set the subpix parameter of the matching cost step"
            " to a value greater than 1 when using dichotomy." in caplog.messages
        )

    @pytest.mark.parametrize("matching_cost_method", ["ssd", "sad", "zncc_python"])
    @pytest.mark.parametrize("pipeline_cfg", ["correct_pipeline_with_ambiguity"])
    def test_check_matching_cost_method_with_ambiguity(
        self, matching_cost_method, pipeline_cfg, pandora2d_machine, caplog, request
    ):
        """
        Check the correlation method used in relation to ambiguity.
        """
        check_pipeline_section(request.getfixturevalue(pipeline_cfg), pandora2d_machine)

        assert (
            "This initial version, available in Pandora2d 1.1.0, should not be used with Pandora measurements"
            "(ssd, sad, zncc_python, mc_cnn). An update in a future version will resolve this issue."
            "In the meantime, it is recommended to filter the confidence_measure map using the validity_mask"
            in caplog.messages
        )

    @pytest.mark.usefixtures("import_plugins")
    @pytest.mark.plugin_tests
    @pytest.mark.skipif(importlib.util.find_spec("mc_cnn") is None, reason="MCCNN plugin not installed")
    @pytest.mark.parametrize("matching_cost_method", ["mc_cnn"])
    @pytest.mark.parametrize("window_size", [11])
    @pytest.mark.parametrize("pipeline_cfg", ["correct_pipeline_with_ambiguity"])
    def test_check_mccnn_with_ambiguity(
        self, matching_cost_method, window_size, pipeline_cfg, pandora2d_machine, caplog, request
    ):
        """
        Check the correlation method used in relation to ambiguity.
        """
        check_pipeline_section(request.getfixturevalue(pipeline_cfg), pandora2d_machine)

        assert (
            "This initial version, available in Pandora2d 1.1.0, should not be used with Pandora measurements"
            "(ssd, sad, zncc_python, mc_cnn). An update in a future version will resolve this issue."
            "In the meantime, it is recommended to filter the confidence_measure map using the validity_mask"
            in caplog.messages
        )


class TestCheckMatchingCostMethodWithImageResampling:
    """Test check_matching_cost_method_with_image_resampling."""

    @pytest.fixture()
    def resampling_type(self):
        return "image"

    @pytest.fixture()
    def pipeline_with_resampling_type(self, correct_pipeline_with_dichotomy_cpp, resampling_type):
        """Cpp dichotomy pipeline with the resampling_type parameter given by the user."""
        correct_pipeline_with_dichotomy_cpp["pipeline"]["refinement"]["resampling_type"] = resampling_type
        return correct_pipeline_with_dichotomy_cpp

    @pytest.mark.parametrize("matching_cost_method", ["ssd", "sad", "zncc_python"])
    def test_fails_with_image_resampling_and_wrong_matching_cost_method(
        self, pipeline_with_resampling_type, pandora2d_machine
    ):
        """Image resampling is not available with every matching cost method."""

        with pytest.raises(ValueError, match="Image resampling is only available with"):
            check_pipeline_section(pipeline_with_resampling_type, pandora2d_machine)

    @pytest.mark.parametrize("matching_cost_method", ["zncc", "zncc-optim-1", "zncc-optim-2", "mutual_information"])
    def test_image_resampling_with_accepted_matching_cost_method(
        self, pipeline_with_resampling_type, pandora2d_machine
    ):
        """It should not fail with image resampling and an accepted matching cost method."""

        check_pipeline_section(pipeline_with_resampling_type, pandora2d_machine)

    @pytest.mark.parametrize("resampling_type", ["cost_surface"])
    @pytest.mark.parametrize("matching_cost_method", ["zncc_python"])
    def test_cost_surface_resampling_with_any_matching_cost_method(
        self, pipeline_with_resampling_type, pandora2d_machine
    ):
        """The matching cost method is only restricted when the dichotomy is performed on the image."""

        check_pipeline_section(pipeline_with_resampling_type, pandora2d_machine)


class TestCheckStep:
    """
    Description : Test check_step.
    Requirement : EX_STEP_02
    """

    def test_nominal_case(self, pipeline_config, pandora2d_machine) -> None:
        """
        Test step configuration with user configuration dictionary
        """

        # Add correct step
        pipeline_config["pipeline"]["matching_cost"]["step"] = [1, 1]
        check_pipeline_section(pipeline_config, pandora2d_machine)

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
        """
        Description : Test check_pipeline_section fails with bad values of step.
        Data :
        Requirement : EX_CONF_08
        """
        pipeline_config["pipeline"]["matching_cost"]["step"] = step
        with pytest.raises(DictCheckerError):
            check_pipeline_section(pipeline_config, pandora2d_machine)
