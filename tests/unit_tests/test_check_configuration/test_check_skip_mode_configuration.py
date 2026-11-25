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
Test check skip mode configuration
"""

import pytest
from json_checker import DictCheckerError

from pandora2d.check_configuration import (
    SCORE_THRESHOLD_DEFAULT_VALUE,
    build_default_skip_rules,
    check_conf,
    check_skip_mode_section,
)


class TestCheckSkipMode:
    """
    Description : Test check_skip_mode_section.
    """

    def test_nominal_case(self, correct_skip_mode) -> None:
        """
        Test function for checking user skip_mode section
        """
        check_skip_mode_section(correct_skip_mode)

    @pytest.mark.parametrize(
        ["parameter", "wrong_value_parameter"],
        [
            pytest.param("type", 12, id="error type value with an int"),
            pytest.param("type", 12.0, id="error type value with a float"),
            pytest.param("type", ["fast"], id="error type value with a list"),
            pytest.param("type", {"value": "fast"}, id="error type value with a dictionary"),
            pytest.param("type", "unknown", id="error type value with an unknown string"),
            pytest.param("rules", 12, id="error rules value with an int"),
            pytest.param("rules", "incomplete_disparity_range", id="error rules value with a string"),
            pytest.param(
                "rules",
                [{"type": "score_threshold", "value": "0.5"}],
                id="error rules value with a string threshold",
            ),
            pytest.param(
                "rules",
                [{"type": "incomplete_disparity_range", "value": 0.5}],
                id="error rules value with an unexpected key",
            ),
            pytest.param(
                "rules",
                [{"type": "incomplete_disparity_range"}, {"type": "incomplete_disparity_range"}],
                id="error rules value with a duplicated rule",
            ),
        ],
    )
    def test_wrong_configuration_raises_exception(self, correct_skip_mode, parameter, wrong_value_parameter):
        """
        Description : Raises an exception if a skip_mode parameter is invalid
        """
        correct_skip_mode["skip_mode"][parameter] = wrong_value_parameter
        with pytest.raises(BaseException):
            check_skip_mode_section(correct_skip_mode)

    @pytest.mark.parametrize(
        "rules",
        [
            pytest.param(None, id="rules absent"),
            pytest.param([], id="rules empty list"),
        ],
    )
    def test_custom_without_rules_raises_exception(self, correct_skip_mode, rules):
        """
        Description : custom requires a non empty rules list (business rule, not schema error)
        """
        if rules is None:
            del correct_skip_mode["skip_mode"]["rules"]
        else:
            correct_skip_mode["skip_mode"]["rules"] = rules

        with pytest.raises(ValueError, match="skip_mode type custom requires rules"):
            check_skip_mode_section(correct_skip_mode)

    @pytest.mark.parametrize("mode_type", ["fast", "complete"])
    def test_empty_rules_raises_exception_for_presets(self, mode_type):
        """
        Description : An empty rules list is rejected for fast and complete (schema error)
        """
        with pytest.raises(ValueError, match="skip_mode rules must not be an empty list"):
            check_skip_mode_section({"skip_mode": {"type": mode_type, "rules": []}})

    def test_update_configuration_without_skip_mode_section(self):
        """
        Description : An absent skip_mode section is not added
        """
        cfg: dict = {}
        check_skip_mode_section(cfg)

        assert cfg == {}

    @pytest.mark.parametrize(
        ["mode_type", "with_rules"],
        [
            pytest.param("fast", False, id="fast without rules"),
            pytest.param("complete", False, id="complete without rules"),
            pytest.param("fast", True, id="fast with incomplete_disparity_range"),
            pytest.param("complete", True, id="complete with incomplete_disparity_range"),
        ],
    )
    def test_skip_mode_section_default(self, mode_type, with_rules):
        """
        Description : fast and complete keep their type and contain the default rule
        """
        cfg = {"skip_mode": {"type": mode_type}}
        if with_rules:
            cfg["skip_mode"]["rules"] = [{"type": "incomplete_disparity_range"}]

        check_skip_mode_section(cfg)

        assert cfg["skip_mode"]["type"] == mode_type
        assert cfg["skip_mode"]["rules"] == build_default_skip_rules(mode_type)

    def test_score_threshold_custom_nominal(self, correct_skip_mode) -> None:
        """score_threshold is accepted in custom mode."""
        correct_skip_mode["skip_mode"] = {
            "type": "custom",
            "rules": [{"type": "score_threshold", "value": 0.5}],
        }
        check_skip_mode_section(correct_skip_mode)
        assert correct_skip_mode["skip_mode"]["rules"] == [{"type": "score_threshold", "value": 0.5}]

    def test_score_threshold_default_value(self) -> None:
        """score_threshold without value gets the default threshold."""
        cfg = {"skip_mode": {"type": "custom", "rules": [{"type": "score_threshold"}]}}
        check_skip_mode_section(cfg)
        assert cfg["skip_mode"]["rules"] == [{"type": "score_threshold", "value": SCORE_THRESHOLD_DEFAULT_VALUE}]

    @pytest.mark.parametrize(
        ["skip_mode_type"],
        [
            pytest.param("fast", id="fast default"),
            pytest.param("complete", id="complete default"),
        ],
    )
    def test_preset_overrides_score_threshold_value(self, skip_mode_type: str) -> None:
        """Preset merges rules and overrides score_threshold value only."""
        cfg = {"skip_mode": {"type": skip_mode_type, "rules": [{"type": "score_threshold", "value": 0.4}]}}
        check_skip_mode_section(cfg)
        default_rules = build_default_skip_rules(skip_mode_type)
        expected_rules = default_rules.copy()
        # Find and update score_threshold value
        for rule in expected_rules:
            if rule["type"] == "score_threshold":
                rule["value"] = 0.4
                break
        assert cfg["skip_mode"]["rules"] == expected_rules


class TestScoreThresholdRequiresRefinement:
    """score_threshold is only valid when the pipeline includes refinement."""

    @pytest.fixture()
    def configuration_with_refinement(self, correct_input_cfg, correct_pipeline_with_dichotomy_cpp):
        return {
            **correct_input_cfg,
            **correct_pipeline_with_dichotomy_cpp,
            "output": {"path": "here"},
        }

    def test_score_threshold_without_refinement_raises(
        self, correct_input_cfg, correct_pipeline_without_refinement, pandora2d_machine
    ):
        configuration = {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            "skip_mode": {"type": "custom", "rules": [{"type": "score_threshold", "value": 0.5}]},
            "output": {"path": "here"},
        }
        with pytest.raises(ValueError, match="score_threshold rule requires a refinement step"):
            check_conf(configuration, pandora2d_machine)

    def test_score_threshold_with_refinement_is_accepted(self, configuration_with_refinement, pandora2d_machine):
        configuration_with_refinement["skip_mode"] = {
            "type": "custom",
            "rules": [{"type": "score_threshold", "value": 0.5}],
        }
        check_conf(configuration_with_refinement, pandora2d_machine)


class TestIncompleteDisparityRangeWithMatchingCost:
    """
    Description : Test that incomplete_disparity_range rule is forbidden with pandora matching cost methods.
    """

    @pytest.fixture()
    def configuration(self, correct_input_cfg, correct_pipeline_without_refinement, correct_skip_mode):
        return {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            **correct_skip_mode,
            "output": {"path": "here"},
        }

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc_python"])
    @pytest.mark.parametrize("mode_type", ["custom", "fast", "complete"])
    def test_pandora_method_raises_exception(self, configuration, pandora2d_machine, mode_type):
        """
        Description : incomplete_disparity_range cannot be used with a pandora matching cost method.
        """
        configuration["skip_mode"]["type"] = mode_type

        with pytest.raises(ValueError, match="incomplete_disparity_range rule is not available"):
            check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("matching_cost_method", ["zncc", "zncc-optim-1", "zncc-optim-2", "mutual_information"])
    def test_cpp_method_is_accepted(self, configuration, pandora2d_machine):
        """
        Description : incomplete_disparity_range can be used with a cpp matching cost method.
        """
        check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc_python"])
    def test_pandora_method_without_skip_mode(self, configuration, pandora2d_machine):
        """
        Description : Without skip_mode section, pandora matching cost methods are accepted.
        """
        del configuration["skip_mode"]

        check_conf(configuration, pandora2d_machine)
