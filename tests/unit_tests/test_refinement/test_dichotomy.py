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
import numpy as np
import pytest
import json_checker

import xarray as xr

from pandora.margins import Margins
from pytest_mock import MockerFixture

from pandora2d.matching_cost import MatchingCost
from pandora2d import refinement


# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name


@pytest.fixture()
def cost_volumes():
    """Pandora2d cost volumes fake data."""
    number_of_rows = 2
    rows = np.arange(number_of_rows)

    number_of_cols = 3
    cols = np.arange(number_of_cols)

    min_disparity_col = -1
    max_disparity_col = 2
    number_of_disparity_col = max_disparity_col - min_disparity_col + 1

    min_disparity_row = 2
    max_disparity_row = 5
    number_of_disparity_row = max_disparity_row - min_disparity_row + 1

    data = np.arange(number_of_rows * number_of_cols * number_of_disparity_col * number_of_disparity_row).reshape(
        (number_of_rows, number_of_cols, number_of_disparity_col, number_of_disparity_row)
    )
    attrs = {"col_to_compute": 1, "sampling_interval": 1}

    return MatchingCost.allocate_cost_volumes(
        attrs,
        rows,
        cols,
        [min_disparity_col, max_disparity_col],
        [min_disparity_row, max_disparity_row],
        data,
    )


@pytest.fixture()
def disp_map():
    """Fake disparity maps."""
    row = np.array(
        [
            [4, 3, 4],
            [3, 4, 3],
        ]
    )
    col = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    return xr.Dataset(
        {
            "row_map": (["row", "col"], row),
            "col_map": (["row", "col"], col),
        },
        coords={
            "row": np.arange(row.shape[0]),
            "col": np.arange(col.shape[1]),
        },
    )


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
    dichotomy_instance.refinement_method(mocker.ANY, mocker.ANY, mocker.ANY, mocker.ANY)

    assert "refinement_method of Dichotomy not yet implemented" in caplog.messages


def test_margins():
    """
    Test margins of Dichotomy.
    """

    config = {"refinement_method": "dichotomy", "iterations": 2, "filter": "sinc"}

    dichotomy_instance = refinement.dichotomy.Dichotomy(config)

    assert dichotomy_instance.margins == Margins(2, 2, 2, 2)


class TestDichotomyWindows:
    """Test DichotomyWindows container."""

    @pytest.mark.parametrize(
        ["row_index", "col_index", "expected"],
        [
            pytest.param(
                0,
                0,
                xr.DataArray(
                    [
                        [5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15],
                    ],
                    coords={
                        "row": 0,
                        "col": 0,
                        "disp_col": [0, 1, 2],
                        "disp_row": [3, 4, 5],
                    },
                    dims=["disp_col", "disp_row"],
                ),
                id="First value",
            ),
            pytest.param(
                1,
                2,
                xr.DataArray(
                    [
                        [80, 81, 82],
                        [84, 85, 86],
                        [88, 89, 90],
                    ],
                    coords={
                        "row": 1,
                        "col": 2,
                        "disp_col": [-1, 0, 1],
                        "disp_row": [2, 3, 4],
                    },
                    dims=["disp_col", "disp_row"],
                ),
                id="Another value",
            ),
        ],
    )
    def test_direct_item_access(self, cost_volumes, disp_map, row_index, col_index, expected):
        """Test we are able to get a dichotomy windows from cost_volumes at given index."""
        disparity_margins = Margins(2, 2, 2, 2)

        dichotomy_windows = refinement.dichotomy.DichotomyWindows(cost_volumes, disp_map, disparity_margins)
        result = dichotomy_windows[row_index, col_index]

        assert result.equals(expected)

    def test_iteration(self, cost_volumes, disp_map):
        """Test we can iterate over dichotomy windows."""
        disparity_margins = Margins(2, 2, 2, 2)

        dichotomy_windows = refinement.dichotomy.DichotomyWindows(cost_volumes, disp_map, disparity_margins)

        result = list(dichotomy_windows)

        assert len(result) == disp_map.sizes["row"] * disp_map.sizes["col"]
        assert result[0].equals(
            xr.DataArray(
                [
                    [5, 6, 7],
                    [9, 10, 11],
                    [13, 14, 15],
                ],
                coords={
                    "row": 0,
                    "col": 0,
                    "disp_col": [0, 1, 2],
                    "disp_row": [3, 4, 5],
                },
                dims=["disp_col", "disp_row"],
            )
        )
        assert result[-2].equals(
            xr.DataArray(
                [
                    [69, 70, 71],
                    [73, 74, 75],
                    [77, 78, 79],
                ],
                coords={
                    "row": 1,
                    "col": 1,
                    "disp_col": [0, 1, 2],
                    "disp_row": [3, 4, 5],
                },
                dims=["disp_col", "disp_row"],
            )
        )
