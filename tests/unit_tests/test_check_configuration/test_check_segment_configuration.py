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
Test check segment step configuration
"""

from typing import Any

import pytest

from pandora2d.check_configuration import build_default_segment_mode_configuration, check_segment_mode_section


class TestCheckSegmentMode:
    """
    Description : Test check_segment_mode_section.
    """

    def test_nominal_case(self, correct_segment_mode) -> None:
        """
        Test function for checking user segment_mode section
        """
        # with a correct segment_mode in check_segment_mode_section should return nothing
        check_segment_mode_section(correct_segment_mode)

    @pytest.mark.parametrize(
        ["parameter", "wrong_value_parameter"],
        [
            pytest.param("enable", 12, id="error enable value with an int"),
            pytest.param("enable", 12.0, id="error enable value with a float"),
            pytest.param("enable", [True, True], id="error enable value with a list of boolean"),
            pytest.param("enable", [1, 1], id="error enable value with a list of int"),
            pytest.param("enable", {"value": True}, id="error enable value with a dictionnary"),
            pytest.param("memory_per_work", 0, id="error memory_per_work value with a zero"),
            pytest.param("memory_per_work", -10, id="error memory_per_work value with a negative number"),
            pytest.param("memory_per_work", 10.12, id="error memory_per_work value with a float"),
            pytest.param("memory_per_work", [True, True], id="error enable memory_per_work with a list of boolean"),
            pytest.param("memory_per_work", [1000, 1000], id="error enable memory_per_work with a list of int"),
            pytest.param("enable", {"value": 2000}, id="error enable value with a dictionnary"),
        ],
    )
    def test_wrong_configuration_raises_exception(self, correct_segment_mode, parameter, wrong_value_parameter):
        """
        Description : Raises an exception if the enable parameter are not a boolean
        """
        correct_segment_mode["segment_mode"][parameter] = wrong_value_parameter
        with pytest.raises(BaseException):
            check_segment_mode_section(correct_segment_mode)

    def test_update_configuration_without_segment_mode_section(self):
        """
        Description : Check that a section is returned specifying that the mode is false if it is not present in the
        user configuration
        """
        cfg: dict[str, dict[Any, Any]] = {}
        check_segment_mode_section(cfg)

        assert cfg == build_default_segment_mode_configuration()

    @pytest.mark.parametrize("enable", [True, False])
    def test_segment_mode_section_default(self, enable):
        """
        Description : Check that a section is returned specifying that the mode is false if it is not present in the
        user configuration
        """
        cfg = {"segment_mode": {"enable": enable}}
        check_segment_mode_section(cfg)

        assert cfg["segment_mode"]["memory_per_work"] == 1000
