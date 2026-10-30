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

from pandora2d.check_configuration import build_default_skip_mode_configuration, check_skip_mode_section


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
                [{"type": "score_threshold", "value": 0.5}],
                id="score_threshold still in development",
                marks=pytest.mark.skip(reason="score_threshold rule is still in development"),
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
        with pytest.raises(DictCheckerError):
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
        assert cfg["skip_mode"]["rules"] == build_default_skip_mode_configuration()["skip_mode"]["rules"]
