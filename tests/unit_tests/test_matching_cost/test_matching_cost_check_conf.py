# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test check_conf method from Matching cost
"""

import importlib.util
import json_checker
import pytest
from pandora.matching_cost import AbstractMatchingCost
from pandora.margins import Margins

from pandora2d import matching_cost

# pylint: disable=redefined-outer-name


def test_check_conf():
    """
    Description : test check_conf of matching cost pipeline
    Data :
    Requirement : EX_MC_ZNCC_00
    """
    matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5})


def test_invalid_method():
    """
    Description : census is not expected to be used with pandora2d.
    Data :
    Requirement : EX_CONF_08
    """
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.MatchingCost({"matching_cost_method": "census", "window_size": 5})


class TestWindowSize:
    """
    Description : Test window_size parameter values.
    Requirement : EX_CONF_04, EX_MC_00
    """

    @pytest.mark.parametrize("method", ["zncc", "sad", "ssd"])
    def test_default_window_size(self, method):
        result = matching_cost.MatchingCost({"matching_cost_method": method, "step": [1, 1]})

        assert result.cfg["window_size"] == AbstractMatchingCost._WINDOW_SIZE  # pylint: disable=W0212 protected-access

    @pytest.mark.parametrize("method", ["zncc", "sad", "ssd"])
    def test_fails_with_invalid_window_size(self, method):
        """
        Description : Test the validity of the window_size parameter
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.MatchingCost({"matching_cost_method": method, "window_size": -1})
        assert "window_size" in err.value.args[0]


@pytest.mark.usefixtures("import_plugins")
@pytest.mark.plugin_tests
@pytest.mark.skipif(importlib.util.find_spec("mc_cnn") is None, reason="MCCNN plugin not installed")
class TestMCCNNConf:
    """
    Description : Test window_size with MCCNN plugin.
    Requirement : EX_CONF_04, EX_MC_00
    """

    def test_default_window_size(self):
        result = matching_cost.MatchingCost({"matching_cost_method": "mc_cnn", "step": [1, 1]})
        assert result.cfg["window_size"] == 11

    def test_fails_with_invalid_window_size(self):
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.MatchingCost({"matching_cost_method": "mc_cnn", "window_size": 5})
        assert "window_size" in err.value.args[0]


class TestStep:
    """
    Description : Test step in matching_cost configuration
    Requirement : EX_CONF_04, EX_STEP_02, EX_MC_01
    """

    def test_nominal_case(self):
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3]})

    def test_default_step(self):
        result = matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5})

        assert result.cfg["step"] == [1, 1]

    def test_fails_with_negative_step(self):
        """
        Description : Test if the step is negative
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [-2, 3]})

    def test_fails_with_one_element_list(self):
        """
        Description : Test fails if the step is a list of one element
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2]})

    def test_fails_with_more_than_two_element_list(self):
        """
        Description : Test fails if the step is a list of more than 2 elements
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3, 4]})

    def test_fails_with_string_element(self):
        """
        Description : Test fails if the step list contains a string element
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": ["2", 3]})


def test_margins():
    """
    test margins of matching cost pipeline
    """
    _matching_cost = matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5})

    assert _matching_cost.margins == Margins(2, 2, 2, 2)
