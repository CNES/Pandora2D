# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2026 CS GROUP France
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

from pandora2d import matching_cost
from pandora2d.margins import Margins

# pylint: disable=redefined-outer-name


def test_check_conf():
    """
    Description : test check_conf of matching cost pipeline
    Data :
    Requirement : EX_MC_ZNCC_00
    """
    matching_cost.PandoraMatchingCostMethods({"matching_cost_method": "zncc_python", "window_size": 5})


def test_check_conf_mutual_information():
    """
    Description : test check_conf of matching cost pipeline with mutual information
    """
    matching_cost.CorrelationMethods({"matching_cost_method": "mutual_information", "window_size": 5})


# /!\ "zncc" currently target "zncc-optim-1"
def test_check_conf_zncc():
    """
    Description : test check_conf of matching cost pipeline with zncc
    """
    matching_cost.CorrelationMethods({"matching_cost_method": "zncc", "window_size": 5})


def test_check_conf_zncc_optim_2():
    """
    Description : test check_conf of matching cost pipeline with zncc (optim-2)
    """
    matching_cost.CorrelationMethods({"matching_cost_method": "zncc-optim-2", "window_size": 5})


def test_invalid_method():
    """
    Description : census is not expected to be used with pandora2d.
    Data :
    Requirement : EX_CONF_08
    """
    with pytest.raises(json_checker.core.exceptions.DictCheckerError):
        matching_cost.PandoraMatchingCostMethods({"matching_cost_method": "census", "window_size": 5})


class TestFactory:
    """
    Test instances of BaseMatchingCost
    """

    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "sad", "ssd"])
    def test_factory_pandora(self, matching_cost_config, matching_cost_object):
        """
        Description : With 'matching_cost_method' equals to 'zncc_python', 'sad', 'ssd',
        we should get a PandoraMatchingCostMethods object.
        """

        matching_cost_instance = matching_cost_object(matching_cost_config)
        assert isinstance(matching_cost_instance, matching_cost.BaseMatchingCost)
        assert isinstance(matching_cost_instance, matching_cost.PandoraMatchingCostMethods)

    # /!\ "zncc" currently target "zncc-optim-1"
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc", "zncc-optim-2"])
    def test_factory(self, matching_cost_config, matching_cost_object):
        """
        Description : With `matching_cost_method` equals to `mutual_information`, `zncc`,
        we should get a CorrelationMethods object.
        """

        matching_cost_instance = matching_cost_object(matching_cost_config)
        assert isinstance(matching_cost_instance, matching_cost.BaseMatchingCost)
        assert isinstance(matching_cost_instance, matching_cost.CorrelationMethods)


# /!\ "zncc" currently target "zncc-optim-1"
@pytest.mark.parametrize(
    "matching_cost_method", ["zncc_python", "sad", "ssd", "mutual_information", "zncc", "zncc-optim-2"]
)
class TestWindowSize:
    """
    Description : Test window_size parameter values.
    Requirement : EX_CONF_04, EX_MC_00
    """

    def test_default_window_size(self, matching_cost_method, matching_cost_object):
        result = matching_cost_object({"matching_cost_method": matching_cost_method, "step": [1, 1]})

        assert result.cfg["window_size"] == AbstractMatchingCost._WINDOW_SIZE  # pylint: disable=W0212 protected-access
        assert result.cfg["window_size"] == result.defaults["window_size"]

    def test_fails_with_invalid_window_size(self, matching_cost_method, matching_cost_object):
        """
        Description : Test the validity of the window_size parameter
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": -1})
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
        result = matching_cost.PandoraMatchingCostMethods({"matching_cost_method": "mc_cnn", "step": [1, 1]})
        assert result.cfg["window_size"] == 11

    def test_fails_with_invalid_window_size(self):
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.PandoraMatchingCostMethods({"matching_cost_method": "mc_cnn", "window_size": 5})
        assert "window_size" in err.value.args[0]


# /!\ "zncc" currently target "zncc-optim-1"
@pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "zncc", "zncc-optim-2"])
class TestStep:
    """
    Description : Test step in matching_cost configuration
    Requirement : EX_CONF_04, EX_STEP_02, EX_MC_01
    """

    def test_nominal_case(self, matching_cost_method, matching_cost_object):
        matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5, "step": [2, 3]})

    def test_default_step(self, matching_cost_method, matching_cost_object):
        result = matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5})

        assert result.cfg["step"] == [1, 1]

    def test_fails_with_negative_step(self, matching_cost_method, matching_cost_object):
        """
        Description : Test if the step is negative
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5, "step": [-2, 3]})

    def test_fails_with_one_element_list(self, matching_cost_method, matching_cost_object):
        """
        Description : Test fails if the step is a list of one element
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5, "step": [2]})

    def test_fails_with_more_than_two_element_list(self, matching_cost_method, matching_cost_object):
        """
        Description : Test fails if the step is a list of more than 2 elements
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5, "step": [2, 3, 4]})

    def test_fails_with_string_element(self, matching_cost_method, matching_cost_object):
        """
        Description : Test fails if the step list contains a string element
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5, "step": ["2", 3]})


# /!\ "zncc" currently target "zncc-optim-1"
@pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "zncc", "zncc-optim-2"])
def test_margins(matching_cost_method, matching_cost_object):
    """
    test margins of matching cost pipeline
    """
    _matching_cost = matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5})

    assert _matching_cost.margins == Margins(2, 2, 2, 2)


class TestFloatPrecision:
    """
    Description : Test float precision in matching_cost configuration
    """

    # /!\ "zncc" currently target "zncc-optim-1"
    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "zncc", "zncc-optim-2"])
    @pytest.mark.parametrize("float_precision", ["float32", "f", "f4"])
    def test_nominal_case(self, matching_cost_method, matching_cost_object, float_precision):
        matching_cost_object(
            {"matching_cost_method": matching_cost_method, "window_size": 5, "float_precision": float_precision}
        )

    # /!\ "zncc" currently target "zncc-optim-1"
    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "zncc", "zncc-optim-2"])
    def test_default_value(self, matching_cost_method, matching_cost_object):
        result = matching_cost_object({"matching_cost_method": matching_cost_method, "window_size": 5})
        assert result.cfg["float_precision"] == "float32"

    # /!\ "zncc" currently target "zncc-optim-1"
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc", "zncc-optim-2"])
    @pytest.mark.parametrize("float_precision", ["float64", "d", "f8"])
    def test_float64_precision(self, matching_cost_method, matching_cost_object, float_precision):
        matching_cost_object(
            {"matching_cost_method": matching_cost_method, "window_size": 5, "float_precision": float_precision}
        )

    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "sad", "ssd"])
    @pytest.mark.parametrize("float_precision", ["float64", "f8", "d"])
    def test_fails_with_float64_precision_and_pandora_methods(
        self, matching_cost_method, matching_cost_object, float_precision
    ):
        # With pandora matching cost methods, only float32 is accepted as float precision.
        # We check that a ValueError is raised if a value equivalent to "float64" or another incorrect value is used.
        with pytest.raises(
            ValueError,
            match="With sad, ssd, mc_cnn, and zncc_python methods, only the float32 type is accepted",
        ):
            matching_cost_object(
                {"matching_cost_method": matching_cost_method, "window_size": 5, "float_precision": float_precision}
            )

    # /!\ "zncc" currently target "zncc-optim-1"
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc", "zncc-optim-2"])
    @pytest.mark.parametrize("float_precision", ["f32", "test", 3])
    def test_fails_with_incorrect_float_precision(self, matching_cost_method, matching_cost_object, float_precision):
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost_object(
                {"matching_cost_method": matching_cost_method, "window_size": 5, "float_precision": float_precision}
            )

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc_python"])
    @pytest.mark.parametrize("float_precision", ["f32", "test", 3])
    def test_fails_with_incorrect_float_precision_and_pandora_methods(
        self, matching_cost_method, matching_cost_object, float_precision
    ):
        # With pandora matching cost methods, a TypeError is raised if the float_precision values is incorrect
        # because it is the 'np.dtype' method that raises the error in the check_conf, not the JSON checker.
        with pytest.raises(TypeError):
            matching_cost_object(
                {"matching_cost_method": matching_cost_method, "window_size": 5, "float_precision": float_precision}
            )
