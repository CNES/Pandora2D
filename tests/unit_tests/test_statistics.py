#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests of the statistics module."""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import warnings
import numpy as np
import pytest

from pandora2d.statistics import compute_statistics, Statistics, Quantiles


class TestStatistics:
    """Test Statistics class."""

    @pytest.fixture()
    def quantiles(self):
        return Quantiles(1.0, 2.0, 3.0, 4.0, 5.0)

    @pytest.fixture()
    def statistics(self, quantiles):
        """Instantiate a Statistics object."""
        return Statistics(mean=10.5, std=5.3, quantiles=quantiles)

    def test_str(self, statistics):
        """Test conversion to str."""
        assert str(statistics) == "Mean: 10.5 ± 5.3"

    def test_to_dict(self, statistics):
        """Test conversion to dict."""
        assert statistics.to_dict() == {
            "mean": 10.5,
            "std": 5.3,
            "minimal_valid_pixel_ratio": 1.0,
            "quantiles": {"p10": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p90": 5.0},
        }


class TestComputeStatistics:
    """Test compute_statistics function."""

    @pytest.mark.parametrize(
        ["data", "invalid_values", "expected"],
        [
            [np.zeros((10, 8)), None, 0],
            [np.array([[0, 0, 0], [20, 20, 20]]), None, 10],
            [np.array([[0, 0, np.nan], [20, 20, np.nan]]), np.nan, 10],
            [np.array([[0, 0, -99], [20, 20, -99]]), -99, 10],
        ],
    )
    def test_mean(self, data, invalid_values, expected):
        """Test mean statistic result."""
        result = compute_statistics(data, invalid_values)

        assert result.mean == expected

    @pytest.mark.parametrize(
        ["data", "invalid_values", "expected"],
        [
            [np.zeros((10, 8)), None, 0],
            [np.array([[10, 10, 10], [20, 20, 20]]), None, 5],
            [np.array([[0, 0, np.nan], [20, 20, np.nan]]), np.nan, 10],
            [np.array([[0, 0, -99], [20, 20, -99]]), -99, 10],
        ],
    )
    def test_std(self, data, invalid_values, expected):
        """Test std statistic result."""
        result = compute_statistics(data, invalid_values)

        assert result.std == expected

    @pytest.mark.parametrize(
        ["data", "invalid_values", "expected"],
        [
            pytest.param(np.zeros((10, 8)), None, 1, id="No invalid"),
            pytest.param(np.array([[-99, -99, -99], [-99, -99, -99]]), -99, 0, id="All invalid"),
            pytest.param(np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]), np.nan, 0, id="All nans"),
            pytest.param(np.array([[20, 30, -99, -99], [-99, -99, -99, -99]]), -99, 0.25, id="Mixed valid/invalid"),
        ],
    )
    def test_minimal_valid_pixel_ratio(self, data, invalid_values, expected):
        """Test std statistic result."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = compute_statistics(data, invalid_values)

            assert result.minimal_valid_pixel_ratio == expected
