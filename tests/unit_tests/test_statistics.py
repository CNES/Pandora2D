#  Copyright (c) 2024. Centre National d'Etudes Spatiales (CNES).
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

import numpy as np
import pytest
import xarray as xr

from pandora2d.statistics import compute_statistics, Statistics


@pytest.fixture()
def data(raw_data):
    """Build DataArray from raw_data."""
    nb_rows, nb_cols = raw_data.shape
    return xr.DataArray(
        data=raw_data,
        coords={"row": np.arange(nb_rows), "col": np.arange(nb_cols)},
        dims=["row", "col"],
    )


class TestStatistics:
    """Test Statistics class."""

    @pytest.fixture()
    def statistics(self):
        """Instantiate a Statistics object."""
        return Statistics(mean=10.5, std=5.3)

    def test_str(self, statistics):
        """Test conversion to str."""
        # TODO: see formatting of number of digits
        assert str(statistics) == "Mean: 10.5 ± 5.3"

    def test_to_dict(self, statistics):
        """Test conversion to dict."""
        assert statistics.to_dict() == {"mean": 10.5, "std": 5.3}


class TestComputeStatistics:
    """Test compute_statistics function."""

    @pytest.mark.parametrize(
        ["raw_data", "invalid_values", "expected"],
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
        ["raw_data", "invalid_values", "expected"],
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
