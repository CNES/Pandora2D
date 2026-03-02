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
Test check expert mode step configuration
"""

from copy import deepcopy

import pytest
from json_checker import DictCheckerError

from pandora2d.check_configuration import check_expert_mode_section


class TestExpertModeSection:
    """
    Description : Test expert_mode_section.
    """

    @pytest.fixture
    def default_configuration(self):
        """Default expert_mode configuration"""
        return {"expert_mode": {"profiling": {}}}

    def test_expert_mode_section_missing_profile_parameter(self):
        """
        Description : Test if profiling section is missing
        Data :
        Requirement :
        """

        with pytest.raises(DictCheckerError, match="profiling"):
            check_expert_mode_section({"expert_mode": {}})

    @pytest.mark.parametrize(
        ["parameter", "wrong_value_parameter"],
        [
            pytest.param("folder_name", 12, id="error folder name with an int"),
            pytest.param("folder_name", ["folder_name"], id="error folder name with a list"),
            pytest.param("folder_name", {"folder_name": "expert_mode"}, id="error folder name with a dict"),
            pytest.param("folder_name", 12.0, id="error folder name with a float"),
        ],
    )
    def test_configuration_expert_mode(self, parameter, wrong_value_parameter, default_configuration):
        """
        Description : Test if wrong parameters are detected
        Data :
        Requirement :
        """
        config = deepcopy(default_configuration)
        config["expert_mode"]["profiling"][parameter] = wrong_value_parameter
        with pytest.raises(DictCheckerError) as err:
            check_expert_mode_section(config)
        assert "folder_name" in err.value.args[0]
