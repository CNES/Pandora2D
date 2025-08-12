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
Test ambiguity cost volume confidence method
"""

import json_checker
import pytest

import xarray as xr

from pandora2d import cost_volume_confidence

# pylint: disable=redefined-outer-name, protected-access


@pytest.fixture()
def ambiguity_cfg():
    return {"confidence_method": "ambiguity"}


@pytest.fixture()
def cost_volume_confidence_object(ambiguity_cfg):
    return cost_volume_confidence.CostVolumeConfidenceRegistry.get(ambiguity_cfg["confidence_method"])


class TestFactory:  # pylint: disable=too-few-public-methods
    """
    Test instances of CostVolumeConfidence
    """

    def test_factory_ambiguity(self, ambiguity_cfg, cost_volume_confidence_object):
        """
        Test instance of CostVolumeConfidence with ambiguity method
        """

        cost_volume_confidence_instance = cost_volume_confidence_object(ambiguity_cfg)

        assert isinstance(cost_volume_confidence_instance, cost_volume_confidence.CostVolumeConfidence)
        assert isinstance(cost_volume_confidence_instance, cost_volume_confidence.Ambiguity)


class TestCheckConf:
    """
    Test check configuration of ambiguity method
    """

    def test_check_conf(self, ambiguity_cfg):
        """
        Test check_conf of ambiguity method
        """
        cost_volume_confidence.Ambiguity(ambiguity_cfg)

    def test_default_values(self, ambiguity_cfg, cost_volume_confidence_object):
        """
        Test default values of ambiguity method
        """

        cost_volume_confidence_instance = cost_volume_confidence_object(ambiguity_cfg)

        assert cost_volume_confidence_instance._method == "ambiguity"
        assert cost_volume_confidence_instance._eta_max == 0.7
        assert cost_volume_confidence_instance._eta_step == 0.01
        assert cost_volume_confidence_instance._normalization is True

    @pytest.mark.parametrize(
        ["wrong_ambiguity_cfg"],
        [
            pytest.param(
                {"confidence_method": "wrong_ambiguity"},
                id="Wrong method name",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 2},
                id="Eta max out of bounds",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": -4},
                id="Eta step out of bounds",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.02, "normalization": "not_bool"},
                id="Normalization is not a boolean",
            ),
        ],
    )
    def test_fails_with_incorrect_cfg(self, cost_volume_confidence_object, wrong_ambiguity_cfg):
        """
        Test that check conf fails when the configuration is incorrect
        """

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            cost_volume_confidence_object(wrong_ambiguity_cfg)

    def test_fails_with_missing_confidence_method_key(self, cost_volume_confidence_object):
        """
        Test that check conf fails when confidence key is missing in the configuration
        """

        with pytest.raises(
            json_checker.core.exceptions.MissKeyCheckerError,
            match="Missing keys in current response: confidence_method",
        ):
            cost_volume_confidence_object({"eta_max": 0.2, "eta_step": 0.02})


class TestConfidencePrediction:
    """
    Test confidence_prediction method
    """

    @pytest.fixture()
    def empty_dataset(self):
        """
        Empty dataset to check that the warning is printed when the confidence_prediction method is called.
        Fixture to be deleted when the ambiguity has been implemented.
        """
        return xr.Dataset()

    def test_confidence_prediction(self, ambiguity_cfg, cost_volume_confidence_object, empty_dataset, caplog):
        """
        Test confidence_prediction method
        """

        cost_volume_confidence_instance = cost_volume_confidence_object(ambiguity_cfg)
        returned_dataset_1, returned_dataset_2 = cost_volume_confidence_instance.confidence_prediction(
            empty_dataset, empty_dataset, empty_dataset, empty_dataset
        )

        assert "The ambiguity method has not yet been implemented" in caplog.messages
        xr.testing.assert_equal(returned_dataset_1, empty_dataset)
        xr.testing.assert_equal(returned_dataset_2, empty_dataset)
