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

import logging
import copy

import numpy as np
import pytest
import json_checker

import xarray as xr

from pandora.margins import Margins
from pytest_mock import MockerFixture

from pandora2d.matching_cost import MatchingCost
from pandora2d import refinement
from pandora2d.interpolation_filter.bicubic import Bicubic


# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name


@pytest.fixture()
def rows():
    return np.arange(2)


@pytest.fixture()
def cols():
    return np.arange(3)


@pytest.fixture()
def min_disparity_row():
    return 2


@pytest.fixture()
def max_disparity_row():
    return 7


@pytest.fixture()
def min_disparity_col():
    return -2


@pytest.fixture()
def max_disparity_col():
    return 3


@pytest.fixture()
def type_measure():
    return "max"


@pytest.fixture()
def zeros_cost_volumes(
    rows,
    cols,
    min_disparity_row,
    max_disparity_row,
    min_disparity_col,
    max_disparity_col,
    type_measure,
):
    """Create a cost_volumes full of zeros."""
    number_of_disparity_col = max_disparity_col - min_disparity_col + 1
    number_of_disparity_row = max_disparity_row - min_disparity_row + 1

    data = np.zeros((rows.size, cols.size, number_of_disparity_col, number_of_disparity_row))
    attrs = {
        "col_disparity_source": [min_disparity_col, max_disparity_col],
        "row_disparity_source": [min_disparity_row, max_disparity_row],
        "col_to_compute": 1,
        "sampling_interval": 1,
        "type_measure": type_measure,
        "step": [1, 1],
    }

    return MatchingCost.allocate_cost_volumes(
        attrs,
        rows,
        cols,
        np.arange(min_disparity_col, max_disparity_col + 1),
        np.arange(min_disparity_row, max_disparity_row + 1),
        data,
    )


@pytest.fixture()
def cost_volumes(zeros_cost_volumes):
    """Pandora2d cost volumes fake data."""
    zeros_cost_volumes["cost_volumes"].data[:] = np.arange(zeros_cost_volumes["cost_volumes"].data.size).reshape(
        zeros_cost_volumes["cost_volumes"].data.shape
    )
    return zeros_cost_volumes


@pytest.fixture()
def invalid_disparity():
    return np.nan


@pytest.fixture()
def disp_map(invalid_disparity, rows, cols):
    """Fake disparity maps with alternating values."""
    row = np.full(rows.size * cols.size, 4.0)
    row[::2] = 5
    col = np.full(rows.size * cols.size, 0.0)
    col[::2] = 1
    return xr.Dataset(
        {
            "row_map": (["row", "col"], row.reshape((rows.size, cols.size))),
            "col_map": (["row", "col"], col.reshape((rows.size, cols.size))),
        },
        coords={
            "row": rows,
            "col": cols,
        },
        attrs={"invalid_disp": invalid_disparity},
    )


@pytest.fixture()
def config():
    """Basic configuration expected to be good."""
    return {"refinement_method": "dichotomy", "iterations": 2, "filter": "bicubic"}


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

    @pytest.mark.parametrize("filter_name", ["bicubic"])
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


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(np.arange(2), id="Row without ROI"),
        pytest.param(np.arange(2, 4), id="Row with ROI"),
        pytest.param(np.arange(0, 6, 2), id="Row without ROI, with step of 2"),
    ],
)
@pytest.mark.parametrize(
    "cols",
    [
        pytest.param(np.arange(3), id="Col without ROI"),
        pytest.param(np.arange(3, 6), id="Col with ROI"),
        pytest.param(np.arange(0, 9, 2), id="Row without ROI, with step of 2"),
    ],
)
class TestRefinementMethod:
    """Test refinement method."""

    @pytest.fixture()
    def cost_volumes(self, zeros_cost_volumes):
        """Build cost volumes."""
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        # but use coordinates for disp_row and disp_col
        zeros_cost_volumes["cost_volumes"].isel(row=0, col=2).loc[{"disp_col": [1, 2], "disp_row": 4}] = [8, 9]
        zeros_cost_volumes["cost_volumes"].isel(row=0, col=2).loc[{"disp_col": [1, 2], "disp_row": 5}] = [10, 8]
        zeros_cost_volumes["cost_volumes"].isel(row=1, col=0).loc[{"disp_col": [-2, -1, 0], "disp_row": 4}] = [
            4.9,
            4.99,
            5,
        ]
        return zeros_cost_volumes

    @pytest.fixture()
    def iterations(self):
        return 1

    @pytest.fixture()
    def filter_name(self):
        return "bicubic"

    @pytest.fixture()
    def config(self, iterations, filter_name):
        return {
            "refinement_method": "dichotomy",
            "iterations": iterations,
            "filter": filter_name,
        }

    @pytest.fixture()
    def dichotomy_instance(self, config):
        return refinement.dichotomy.Dichotomy(config)

    @pytest.mark.parametrize(["iterations", "precision"], [[1, 0.5], [2, 0.25], [3, 0.125]])
    def test_precision_is_logged(
        self, cost_volumes, disp_map, dichotomy_instance, precision, mocker: MockerFixture, caplog
    ):
        """Precision should be logged."""
        with caplog.at_level(logging.INFO):
            dichotomy_instance.refinement_method(cost_volumes, disp_map, img_left=mocker.ANY, img_right=mocker.ANY)
        assert caplog.record_tuples == [("root", logging.INFO, f"Dichotomy precision reached: {precision}")]

    @pytest.mark.parametrize(
        ["type_measure", "expected"],
        [
            pytest.param("min", np.nanargmin, id="min"),
            pytest.param("max", np.nanargmax, id="max"),
        ],
    )
    def test_which_cost_selection_method_is_used(
        self, dichotomy_instance, cost_volumes, disp_map, type_measure, expected, mocker: MockerFixture
    ):
        """Test cost_volume's type_measure attrs determines which cost_selection_method is used."""
        cost_volumes.attrs["type_measure"] = type_measure
        mocked_search_new_best_point = mocker.patch(
            "pandora2d.refinement.dichotomy.search_new_best_point",
            autospec=True,
            return_value=(refinement.dichotomy.Point(0, 0), 0, 0, 0),
        )

        dichotomy_instance.refinement_method(cost_volumes, disp_map, img_left=mocker.ANY, img_right=mocker.ANY)

        mocked_search_new_best_point.assert_called_with(
            cost_surface=mocker.ANY,
            precision=mocker.ANY,
            initial_disparity=mocker.ANY,
            initial_position=mocker.ANY,
            initial_value=mocker.ANY,
            filter_dicho=mocker.ANY,
            cost_selection_method=expected,
        )

    def test_result_of_one_iteration(self, dichotomy_instance, cost_volumes, disp_map, mocker: MockerFixture):
        """Test result of refinement method is as expected."""

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, img_left=mocker.ANY, img_right=mocker.ANY
        )

        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_row[1, 0] == disp_map["row_map"][1, 0]
        assert result_disp_col[1, 0] == disp_map["col_map"][1, 0] - 0.5

    @pytest.mark.parametrize("iterations", [2])
    def test_result_of_two_iterations(self, dichotomy_instance, cost_volumes, disp_map, mocker: MockerFixture):
        """Test result of refinement method is as expected."""

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, img_left=mocker.ANY, img_right=mocker.ANY
        )

        # Different results from the spline filter
        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.25
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.25
        assert result_disp_row[1, 0] == disp_map["row_map"][1, 0]
        assert result_disp_col[1, 0] == disp_map["col_map"][1, 0] - 0.25

    def test_with_nans_in_disp_map(self, dichotomy_instance, cost_volumes, disp_map, mocker: MockerFixture):
        """Test that even with NaNs in disparity maps we can extract values from cost_volumes."""
        # Convert disp_map to float so that it can store NaNs:
        disp_map = disp_map.astype(np.float32)
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map[{"row": 1, "col": 0}] = np.nan

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, img_left=mocker.ANY, img_right=mocker.ANY
        )

        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert np.isnan(result_disp_row[1, 0])
        assert np.isnan(result_disp_col[1, 0])

    @pytest.mark.parametrize("invalid_disparity", [-9999])
    def test_with_invalid_values_in_disp_map(
        self, dichotomy_instance, cost_volumes, disp_map, invalid_disparity, mocker: MockerFixture
    ):
        """Test that even with invalid values in disparity maps we can extract other values from cost_volumes."""
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map["row_map"][{"row": 1, "col": 0}] = invalid_disparity
        disp_map["col_map"][{"row": 0, "col": 1}] = invalid_disparity

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes,
            copy_disp_map,
            img_left=mocker.ANY,
            img_right=mocker.ANY,
        )

        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert result_disp_row[1, 0] == invalid_disparity
        assert result_disp_col[0, 1] == invalid_disparity
        assert result_disp_row[0, 1] != invalid_disparity
        assert result_disp_col[1, 0] != invalid_disparity

    def test_disparity_map_is_within_range(  # pylint: disable=too-many-arguments
        self,
        dichotomy_instance,
        cost_volumes,
        disp_map,
        min_disparity_row,
        min_disparity_col,
        max_disparity_row,
        max_disparity_col,
        mocker: MockerFixture,
    ):
        """Resulting disparity should not be outside of initial disparity range."""
        # Setting points to extreme disparity values will let dichotomy find values outside of range
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map["row_map"][{"row": 0, "col": 0}] = min_disparity_row
        disp_map["col_map"][{"row": 0, "col": 0}] = min_disparity_col
        disp_map["row_map"][{"row": 1, "col": 0}] = max_disparity_row
        disp_map["col_map"][{"row": 1, "col": 0}] = max_disparity_col
        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes,
            disp_map,
            img_left=mocker.ANY,
            img_right=mocker.ANY,
        )

        assert np.nanmin(result_disp_row) >= min_disparity_row
        assert np.nanmin(result_disp_col) >= min_disparity_col
        assert np.nanmax(result_disp_row) <= max_disparity_row
        assert np.nanmax(result_disp_col) <= max_disparity_col


def test_margins():
    """
    Test margins of Dichotomy.
    """

    config = {"refinement_method": "dichotomy", "iterations": 2, "filter": "bicubic"}

    dichotomy_instance = refinement.dichotomy.Dichotomy(config)

    assert dichotomy_instance.margins == Margins(1, 1, 2, 2)


class TestCostSurfaces:
    """Test CostSurfaces container."""

    @pytest.mark.parametrize(
        ["row_index", "col_index", "expected"],
        [
            pytest.param(
                0,
                0,
                xr.DataArray(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ],
                    coords={
                        "row": 0,
                        "col": 0,
                        "disp_col": [-2, -1, 0, 1, 2, 3],
                        "disp_row": [2, 3, 4, 5, 6, 7],
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
                        [180, 181, 182, 183, 184, 185],
                        [186, 187, 188, 189, 190, 191],
                        [192, 193, 194, 195, 196, 197],
                        [198, 199, 200, 201, 202, 203],
                        [204, 205, 206, 207, 208, 209],
                        [210, 211, 212, 213, 214, 215],
                    ],
                    coords={
                        "row": 1,
                        "col": 2,
                        "disp_col": [-2, -1, 0, 1, 2, 3],
                        "disp_row": [2, 3, 4, 5, 6, 7],
                    },
                    dims=["disp_col", "disp_row"],
                ),
                id="Another value",
            ),
        ],
    )
    def test_direct_item_access(self, cost_volumes, row_index, col_index, expected):
        """Test we are able to get a dichotomy windows from cost_volumes at given index."""

        cost_surfaces = refinement.dichotomy.CostSurfaces(cost_volumes)
        result = cost_surfaces[row_index, col_index]

        assert result.equals(expected)

    def test_cost_volumes_dimensions_order(self, cost_volumes):
        """This test is here to show that disp_row is along columns numpy array and disp_col along numpy array rows."""

        cost_surfaces = refinement.dichotomy.CostSurfaces(cost_volumes)
        result = cost_surfaces[0, 0]

        # Result is:
        # xr.DataArray(
        #     [
        #       [  0,   1,   2,   3,   4,   5],
        #       [  6,   7,   8,   9,  10,  11],
        #       [ 12,  13,  14,  15,  16,  17],
        #       [ 18,  19,  20,  21,  22,  23],
        #       [ 24,  25,  26,  27,  28,  29],
        #       [ 30,  31,  32,  33,  34,  35],
        # ],
        #     coords={
        #         "row": 0,
        #         "col": 0,
        #         "disp_col": [-2, -1, 0, 1, 2, 3],
        #         "disp_row": [2, 3, 4, 5, 6, 7],
        #     },
        #     dims=["disp_col", "disp_row"],
        # )

        # disp_row is along columns numpy array and disp_cal along numpy array rows:
        np.testing.assert_array_equal(result.sel(disp_row=3).data, [1, 7, 13, 19, 25, 31])
        np.testing.assert_array_equal(result.sel(disp_col=0).data, [12, 13, 14, 15, 16, 17])

    def test_iteration(self, cost_volumes, disp_map):
        """Test we can iterate over cost surfaces."""

        cost_surfaces = refinement.dichotomy.CostSurfaces(cost_volumes)

        result = list(cost_surfaces)

        assert len(result) == disp_map.sizes["row"] * disp_map.sizes["col"]
        assert result[0].equals(
            xr.DataArray(
                [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 23],
                    [24, 25, 26, 27, 28, 29],
                    [30, 31, 32, 33, 34, 35],
                ],
                coords={
                    "row": 0,
                    "col": 0,
                    "disp_col": [-2, -1, 0, 1, 2, 3],
                    "disp_row": [2, 3, 4, 5, 6, 7],
                },
                dims=["disp_col", "disp_row"],
            ),
        )
        assert result[-2].equals(
            xr.DataArray(
                [
                    [144, 145, 146, 147, 148, 149],
                    [150, 151, 152, 153, 154, 155],
                    [156, 157, 158, 159, 160, 161],
                    [162, 163, 164, 165, 166, 167],
                    [168, 169, 170, 171, 172, 173],
                    [174, 175, 176, 177, 178, 179],
                ],
                coords={
                    "row": 1,
                    "col": 1,
                    "disp_col": [-2, -1, 0, 1, 2, 3],
                    "disp_row": [2, 3, 4, 5, 6, 7],
                },
                dims=["disp_col", "disp_row"],
            )
        )


@pytest.mark.parametrize(
    ["cost_surface", "precision", "initial_disparity", "initial_position", "initial_value", "expected"],
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.0, 2.0), np.float32(2.0), np.float32(2.0), np.float32(1.0)),
            id="Initial is best",
        ),
        # This case is not realistic as the initial value should be the maximum value, but it is easier to test.
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 20, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(1.5, 2.5), np.float32(1.5), np.float32(2.5), np.float32(6.64453125)),
            id="Bottom left is best",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 20, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.25,
            (1.5, 2.5),
            (1.5, 2.5),
            7.638916,
            (refinement.dichotomy.Point(1.25, 2.75), np.float32(1.25), np.float32(2.75), np.float32(15.09161376953125)),
            id="Bottom left is best at 0.25 precision",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, np.nan, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 20, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            (2, 2),
            (2, 2),
            np.float32(1.0),
            (refinement.dichotomy.Point(2.0, 2.0), np.float32(2.0), np.float32(2.0), np.float32(1.0)),
            id="NaN in kernel gives initial position",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 20, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 2.5), np.float32(2.5), np.float32(2.5), np.float32(6.64453125)),
            id="Bottom right is best",
        ),
        pytest.param(
            np.array(
                [
                    [np.nan, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 20, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 2.5), np.float32(2.5), np.float32(2.5), np.float32(6.64453125)),
            id="NaN outside of kernel has no effect",
        ),
    ],
)
def test_search_new_best_point(cost_surface, precision, initial_disparity, initial_position, initial_value, expected):
    """Test we get new coordinates as expected."""

    filter_dicho = Bicubic("bicubic")

    cost_selection_method = np.nanargmax

    result = refinement.dichotomy.search_new_best_point(
        cost_surface,
        precision,
        initial_disparity,
        initial_position,
        initial_value,
        filter_dicho,
        cost_selection_method,
    )

    assert result == expected
