# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
"""
Test the refinement.dichotomy module.
"""
import pytest
import json_checker

from pandora.margins import Margins
from pytest_mock import MockerFixture

from pandora2d import refinement

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name


@pytest.fixture()
def config():
    """Basic configuration expected to be good."""
    return {"refinement_method": "dichotomy", "iterations": 2, "filter": "sinc"}


def test_factory(config):
    """With `refinement_method` equals to `dichotomy`, we should get a Dichotomy object."""
    dichotomy_instance = refinement.AbstractRefinement(config)  # type: ignore[abstract]

    assert isinstance(dichotomy_instance, refinement.dichotomy.Dichotomy)
    assert isinstance(dichotomy_instance, refinement.AbstractRefinement)


class TestCheckConf:
    """Test the check_conf method."""

    def test_method_field(self, config):
        """An exception should be raised if `refinement_method` is not `dichotomy`."""
        config["refinement_method"] = "invalid_method"

        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            refinement.dichotomy.Dichotomy(config)
        assert "invalid_method" in err.value.args[0]

    def test_iterations_below_minimum(self, config):
        """An exception should be raised."""
        config["iterations"] = 0

        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            refinement.dichotomy.Dichotomy(config)
        assert "Not valid data" in err.value.args[0]
        assert "iterations" in err.value.args[0]

    def test_iterations_above_maximum(self, config, caplog):
        """Test that when user set an iteration value above defined maximum,
        we replace it by this maximum and log a warning.
        """
        config["iterations"] = 10

        dichotomy_instance = refinement.dichotomy.Dichotomy(config)

        assert dichotomy_instance.cfg["iterations"] == 9
        assert (
            "number_of_iterations 10 is above maximum iteration. Maximum value of 9 will be used instead."
            in caplog.messages
        )

    @pytest.mark.parametrize("iterations", [1, 9])
    def test_iterations_in_allowed_range(self, config, iterations):
        """It should not fail."""
        config["iterations"] = iterations

        dichotomy_instance = refinement.dichotomy.Dichotomy(config)

        assert dichotomy_instance.cfg["iterations"] == iterations

    @pytest.mark.parametrize("filter_name", ["sinc", "bicubic", "spline"])
    def test_valid_filter_names(self, config, filter_name):
        """Test accepted filter names."""
        config["filter"] = filter_name

        dichotomy_instance = refinement.dichotomy.Dichotomy(config)

        assert dichotomy_instance.cfg["filter"] == filter_name

    @pytest.mark.parametrize("missing", ["refinement_method", "iterations", "filter"])
    def test_fails_on_missing_keys(self, config, missing):
        """Should raise an error when a mandatory key is missing."""
        del config[missing]

        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError) as err:
            refinement.dichotomy.Dichotomy(config)
        assert f"Missing keys in current response: {missing}" in err.value.args[0]

    def test_fails_on_unexpected_key(self, config):
        """Should raise an error when an unexpected key is given."""
        config["unexpected_key"] = "unexpected_value"

        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError) as err:
            refinement.dichotomy.Dichotomy(config)
        assert "Missing keys in expected schema: unexpected_key" in err.value.args[0]


def test_refinement_method(config, caplog, mocker: MockerFixture):
    """Not yet implemented."""

    dichotomy_instance = refinement.dichotomy.Dichotomy(config)

    # We can pass anything as it is not yet implemented
    dichotomy_instance.refinement_method(mocker.ANY, mocker.ANY)

    assert "refinement_method of Dichotomy not yet implemented" in caplog.messages


def test_margins():
    """
    Test margins of Dichotomy.
    """

    config = {"refinement_method": "dichotomy", "iterations": 2, "filter": "sinc"}

    dichotomy_instance = refinement.dichotomy.Dichotomy(config)

    assert dichotomy_instance.margins == Margins(2, 2, 2, 2)
