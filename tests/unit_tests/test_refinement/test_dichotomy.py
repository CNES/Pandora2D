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
"""
Test the refinement.dichotomy module.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments

import logging
import copy

import numpy as np
import pytest
import json_checker

import xarray as xr

from pandora.margins import Margins
from pytest_mock import MockerFixture

from pandora2d import refinement
from pandora2d.interpolation_filter.bicubic import BicubicPython
from pandora2d.interpolation_filter.bicubic_cpp import Bicubic


def test_factory(dichotomy_python_instance, dichotomy_cpp_instance):
    """
    Description : With `refinement_method` equals to `dichotomy`, we should get a DichotomyPython object.
    Data :
    Requirement : EX_REF_DICH_00
    """
    assert isinstance(dichotomy_python_instance, refinement.dichotomy.DichotomyPython)
    assert isinstance(dichotomy_python_instance, refinement.AbstractRefinement)

    assert isinstance(dichotomy_cpp_instance, refinement.dichotomy_cpp.Dichotomy)
    assert isinstance(dichotomy_cpp_instance, refinement.AbstractRefinement)


class TestCheckConf:
    """
    Description : Test the check_conf method.
    Requirement : EX_CONF_08, EX_REF_01, EX_REF_DICH_01
    """

    @pytest.mark.parametrize(
        ["wrong_refinement_method_name", "dichotomy_class"],
        [
            pytest.param("invalid_name", refinement.dichotomy.DichotomyPython),
            pytest.param("invalid_name", refinement.dichotomy_cpp.Dichotomy),
        ],
    )
    def test_method_field(self, config_dichotomy, wrong_refinement_method_name, dichotomy_class):
        """An exception should be raised if `refinement_method` is not `dichotomy`."""

        config_dichotomy["refinement_method"] = wrong_refinement_method_name

        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            dichotomy_class(config_dichotomy)
        assert "invalid_name" in err.value.args[0]

    @pytest.mark.parametrize("iterations", [0])
    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_iterations_below_minimum(self, config_dichotomy, dichotomy_class, dichotomy_class_str):
        """An exception should be raised."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            dichotomy_class(config_dichotomy)
        assert "Not valid data" in err.value.args[0]
        assert "iterations" in err.value.args[0]

    @pytest.mark.parametrize("iterations", [10])
    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_iterations_above_maximum(self, config_dichotomy, caplog, dichotomy_class, dichotomy_class_str):
        """Test that when user set an iteration value above defined maximum,
        we replace it by this maximum and log a warning.
        """

        config_dichotomy["refinement_method"] = dichotomy_class_str
        # caplog does not capture logs from fixture, so we can not use dichotomy_python fixture
        dichotomy_instance = dichotomy_class(config_dichotomy)

        assert dichotomy_instance.cfg["iterations"] == 9
        assert (
            "number_of_iterations 10 is above maximum iteration. Maximum value of 9 will be used instead."
            in caplog.messages
        )

    @pytest.mark.parametrize("iterations", [1, 9])
    @pytest.mark.parametrize(
        "dichotomy_instance_name",
        ["dichotomy_python_instance", "dichotomy_cpp_instance"],
    )
    def test_iterations_in_allowed_range(self, request, iterations, dichotomy_instance_name):
        """It should not fail."""
        dichotomy_instance = request.getfixturevalue(dichotomy_instance_name)
        assert dichotomy_instance.cfg["iterations"] == iterations

    @pytest.mark.parametrize(
        ["dichotomy_instance_name", "filter_name"],
        [
            ("dichotomy_python_instance", "bicubic"),
            ("dichotomy_python_instance", "sinc_python"),
            ("dichotomy_cpp_instance", "bicubic"),
            ("dichotomy_cpp_instance", "sinc"),
        ],
    )
    def test_valid_filter_names(self, request, config_dichotomy, dichotomy_instance_name):
        """
        Description : Test accepted filter names.
        Data :
        Requirement :
               * EX_REF_BCO_00
               * EX_REF_SINC_00
        """
        dichotomy_instance = request.getfixturevalue(dichotomy_instance_name)
        assert dichotomy_instance.cfg["filter"] == config_dichotomy["filter"]

    @pytest.mark.parametrize(
        ["config_dict", "dichotomy_class"],
        [
            pytest.param(
                {
                    "refinement_method": "dichotomy_python",
                    "iterations": 1,
                    "filter": {"method": "sinc_python", "size": 42},
                },
                refinement.dichotomy.DichotomyPython,
                id="sinc_python",
            ),
            pytest.param(
                {
                    "refinement_method": "dichotomy",
                    "iterations": 1,
                    "filter": {"method": "sinc", "size": 42},
                },
                refinement.dichotomy_cpp.Dichotomy,
                id="sinc",
            ),
        ],
    )
    def test_fails_with_bad_filter_configuration(self, config_dict, dichotomy_class):
        """Test accepted filter names."""
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            dichotomy_class(config_dict)
        assert "size" in err.value.args[0]

    @pytest.mark.parametrize("filter_name", ["invalid_name"])
    @pytest.mark.parametrize(
        ["dichotomy_class", "dichotomy_class_str"],
        [
            pytest.param(refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            pytest.param(refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_faild_with_invalid_filter_name(self, config_dichotomy, dichotomy_class, dichotomy_class_str):
        """Should raise an error when filter has invalid name."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            dichotomy_class(config_dichotomy)
        assert "filter" in err.value.args[0]

    @pytest.mark.parametrize("missing", ["refinement_method", "iterations", "filter"])
    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_fails_on_missing_keys(self, config_dichotomy, missing, dichotomy_class, dichotomy_class_str):
        """
        Description : Should raise an error when a mandatory key is missing.
        Data :
        Requirement : EX_CONF_08
        """
        config_dichotomy["refinement_method"] = dichotomy_class_str
        del config_dichotomy[missing]

        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError) as err:
            dichotomy_class(config_dichotomy)
        assert f"Missing keys in current response: {missing}" in err.value.args[0]

    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_fails_on_unexpected_key(self, config_dichotomy, dichotomy_class, dichotomy_class_str):
        """Should raise an error when an unexpected key is given."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        config_dichotomy["unexpected_key"] = "unexpected_value"

        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError) as err:
            dichotomy_class(config_dichotomy)
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

    @pytest.mark.parametrize("subpixel", [1, 2])
    @pytest.mark.parametrize(["iterations", "precision"], [[1, 0.5], [2, 0.25], [3, 0.125]])
    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_precision_is_logged(
        self,
        cost_volumes,
        disp_map,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        precision,
        left_img,
        mocker: MockerFixture,
        caplog,
    ):
        """
        Description : Precision should be logged.
        Data :
        Requirement : EX_REF_DICH_01
        """
        config_dichotomy["refinement_method"] = dichotomy_class_str
        # caplog does not capture logs from fixture, so we can not use dichotomy_python fixture
        dichotomy_instance = dichotomy_class(config_dichotomy)
        with caplog.at_level(logging.INFO):
            dichotomy_instance.refinement_method(cost_volumes, disp_map, left_img, img_right=mocker.ANY)
        assert ("root", logging.INFO, f"Dichotomy precision reached: {precision}") in caplog.record_tuples

    @pytest.mark.parametrize(
        ["type_measure", "expected"],
        [
            pytest.param("min", np.nanargmin, id="min"),
            pytest.param("max", np.nanargmax, id="max"),
        ],
    )
    def test_which_cost_selection_method_is_used(
        self, dichotomy_python_instance, cost_volumes, disp_map, left_img, type_measure, expected, mocker: MockerFixture
    ):
        """Test cost_volume's type_measure attrs determines which cost_selection_method is used."""
        cost_volumes.attrs["type_measure"] = type_measure
        mocked_search_new_best_point = mocker.patch(
            "pandora2d.refinement.dichotomy.search_new_best_point",
            autospec=True,
            return_value=(refinement.dichotomy.Point(0, 0), 0, 0, 0),
        )

        dichotomy_python_instance.refinement_method(cost_volumes, disp_map, left_img, img_right=mocker.ANY)

        mocked_search_new_best_point.assert_called_with(
            cost_surface=mocker.ANY,
            precision=mocker.ANY,
            initial_disparity=mocker.ANY,
            initial_position=mocker.ANY,
            initial_value=mocker.ANY,
            filter_dicho=mocker.ANY,
            cost_selection_method=expected,
        )

    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_result_of_one_iteration(
        self,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        cost_volumes,
        disp_map,
        left_img,
        mocker: MockerFixture,
    ):
        """Test result of refinement method is as expected."""

        config_dichotomy["refinement_method"] = dichotomy_class_str
        dichotomy_instance = dichotomy_class(config_dichotomy)
        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_img, img_right=mocker.ANY
        )

        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_row[1, 0] == disp_map["row_map"][1, 0]
        assert result_disp_col[1, 0] == disp_map["col_map"][1, 0] - 0.5

    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    @pytest.mark.parametrize("iterations", [2])
    def test_result_of_two_iterations(
        self,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        cost_volumes,
        left_img,
        disp_map,
        mocker: MockerFixture,
    ):
        """Test result of refinement method is as expected."""

        config_dichotomy["refinement_method"] = dichotomy_class_str
        dichotomy_instance = dichotomy_class(config_dichotomy)
        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_img, img_right=mocker.ANY
        )

        # Different results from the spline filter
        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.25
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.25
        assert result_disp_row[1, 0] == disp_map["row_map"][1, 0]
        assert result_disp_col[1, 0] == disp_map["col_map"][1, 0] - 0.25

    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_with_nans_in_disp_map(
        self,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        cost_volumes,
        disp_map,
        left_img,
        mocker: MockerFixture,
    ):
        """Test that even with NaNs in disparity maps we can extract values from cost_volumes."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        dichotomy_instance = dichotomy_class(config_dichotomy)
        # Convert disp_map to float so that it can store NaNs:
        disp_map = disp_map.astype(cost_volumes["cost_volumes"].data.dtype)
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map[{"row": 1, "col": 0}] = np.nan

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_img, img_right=mocker.ANY
        )

        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert np.isnan(result_disp_row[1, 0])
        assert np.isnan(result_disp_col[1, 0])

    @pytest.mark.parametrize("invalid_disparity", [-9999])
    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_with_invalid_values_in_disp_map(
        self,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        cost_volumes,
        disp_map,
        left_img,
        invalid_disparity,
        mocker: MockerFixture,
    ):
        """Test that even with invalid values in disparity maps we can extract other values from cost_volumes."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        dichotomy_instance = dichotomy_class(config_dichotomy)
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map["row_map"][{"row": 1, "col": 0}] = invalid_disparity
        disp_map["col_map"][{"row": 0, "col": 1}] = invalid_disparity

        copy_disp_map = copy.deepcopy(disp_map)

        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes,
            copy_disp_map,
            left_img,
            img_right=mocker.ANY,
        )

        assert result_disp_row[0, 2] == disp_map["row_map"][0, 2] - 0.5
        assert result_disp_col[0, 2] == disp_map["col_map"][0, 2] + 0.5
        assert result_disp_row[1, 0] == invalid_disparity
        assert result_disp_col[0, 1] == invalid_disparity
        assert result_disp_row[0, 1] != invalid_disparity
        assert result_disp_col[1, 0] != invalid_disparity

    @pytest.mark.parametrize(
        "dichotomy_class, dichotomy_class_str",
        [
            (refinement.dichotomy.DichotomyPython, "dichotomy_python"),
            (refinement.dichotomy_cpp.Dichotomy, "dichotomy"),
        ],
    )
    def test_disparity_map_is_within_range(  # pylint: disable=too-many-arguments
        self,
        dichotomy_class,
        dichotomy_class_str,
        config_dichotomy,
        cost_volumes,
        disp_map,
        left_img,
        min_disparity_row,
        min_disparity_col,
        max_disparity_row,
        max_disparity_col,
        mocker: MockerFixture,
    ):
        """Resulting disparity should not be outside of initial disparity range."""
        config_dichotomy["refinement_method"] = dichotomy_class_str
        dichotomy_instance = dichotomy_class(config_dichotomy)
        # Setting points to extreme disparity values will let dichotomy find values outside of range
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        disp_map["row_map"][{"row": 0, "col": 0}] = min_disparity_row
        disp_map["col_map"][{"row": 0, "col": 0}] = min_disparity_col
        disp_map["row_map"][{"row": 1, "col": 0}] = max_disparity_row
        disp_map["col_map"][{"row": 1, "col": 0}] = max_disparity_col
        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes,
            disp_map,
            left_img,
            img_right=mocker.ANY,
        )

        assert np.nanmin(result_disp_row) >= min_disparity_row
        assert np.nanmin(result_disp_col) >= min_disparity_col
        assert np.nanmax(result_disp_row) <= max_disparity_row
        assert np.nanmax(result_disp_col) <= max_disparity_col

    @pytest.mark.parametrize(
        ["subpixel", "iterations", "nb_of_skipped"],
        [
            pytest.param(2, 1, 1),
            pytest.param(4, 1, 2),
            pytest.param(4, 2, 2),
            pytest.param(8, 3, 3),
        ],
    )
    def test_skip_iterations_with_subpixel(  # pylint: disable=too-many-arguments
        self,
        dichotomy_python_instance,
        cost_volumes,
        disp_map,
        left_img,
        subpixel,
        nb_of_skipped,
        caplog,
        mocker: MockerFixture,
    ):
        """First iterations must be skipped since precision is already reached by subpixel."""
        result_disp_map = copy.deepcopy(disp_map)
        with caplog.at_level(logging.INFO):
            result_disp_col, result_disp_row, _ = dichotomy_python_instance.refinement_method(
                cost_volumes,
                result_disp_map,
                img_left=left_img,
                img_right=mocker.ANY,
            )

        np.testing.assert_array_equal(result_disp_row, disp_map["row_map"])
        np.testing.assert_array_equal(result_disp_col, disp_map["col_map"])
        assert (
            f"With subpixel of `{subpixel}` the `{nb_of_skipped}` first dichotomy iterations will be skipped."
            in caplog.messages
        )


@pytest.mark.parametrize(
    ["filter_name", "iterations", "expected"],
    [
        pytest.param("sinc_python", 1, [0, 0.5], id="sinc_python - 1 iteration"),
        pytest.param("sinc_python", 2, [0, 0.25, 0.5, 0.75], id="sinc_python - 2 iteration"),
    ],
)
def test_pre_computed_filter_fractional_shifts(dichotomy_python_instance, expected):
    """Test filter.fractional_shifts is consistent with dichotomy iteration number."""
    np.testing.assert_array_equal(dichotomy_python_instance.filter.fractional_shifts, expected)


@pytest.mark.parametrize(
    ["refinement_method", "dichotomy_class"],
    [
        pytest.param("dichotomy_python", refinement.dichotomy.DichotomyPython),
        pytest.param("dichotomy", refinement.dichotomy_cpp.Dichotomy),
    ],
)
@pytest.mark.parametrize(
    ["iterations", "filter_cfg", "expected"],
    [
        pytest.param(1, {"method": "bicubic"}, Margins(1, 1, 2, 2)),
        pytest.param(
            1,
            {"method": "sinc_python", "size": 7},
            Margins(7, 7, 7, 7),
        ),
    ],
)
def test_margins(refinement_method, dichotomy_class, iterations, filter_cfg, expected):
    """
    Test margins of DichotomyPython.
    """
    config_dict = {
        "refinement_method": refinement_method,
        "iterations": iterations,
        "filter": filter_cfg,
    }

    assert dichotomy_class(config_dict).margins == expected


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
                        "disp_row": [2, 3, 4, 5, 6, 7],
                        "disp_col": [-2, -1, 0, 1, 2, 3],
                    },
                    dims=["disp_row", "disp_col"],
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
                        "disp_row": [2, 3, 4, 5, 6, 7],
                        "disp_col": [-2, -1, 0, 1, 2, 3],
                    },
                    dims=["disp_row", "disp_col"],
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
        """This test is here to show that disp_row is along rows numpy array and disp_col along numpy array columns ."""

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

        # disp_row is along rows numpy array and disp_cal along numpy array columns:
        np.testing.assert_array_equal(result.sel(disp_row=3).data, [6, 7, 8, 9, 10, 11])
        np.testing.assert_array_equal(result.sel(disp_col=0).data, [2, 8, 14, 20, 26, 32])

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
                    "disp_row": [2, 3, 4, 5, 6, 7],
                    "disp_col": [-2, -1, 0, 1, 2, 3],
                },
                dims=["disp_row", "disp_col"],
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
                    "disp_row": [2, 3, 4, 5, 6, 7],
                    "disp_col": [-2, -1, 0, 1, 2, 3],
                },
                dims=["disp_row", "disp_col"],
            )
        )


@pytest.fixture()
def make_cost_surface(cost_surface_data, subpix):
    """
    Creates a cost surface data array according to given data and subpix
    """

    cost_surface = xr.DataArray(cost_surface_data)

    cost_surface.attrs["subpixel"] = subpix

    return cost_surface


@pytest.mark.parametrize(
    [
        "filter_dicho",
    ],
    [
        pytest.param(
            BicubicPython({"method": "bicubic_python"}),
            id="Bicubic python",
        ),
        pytest.param(
            Bicubic({"method": "bicubic"}),
            id="Bicubic cpp",
        ),
    ],
)
@pytest.mark.parametrize(
    [
        "cost_surface_data",
        "subpix",
        "precision",
        "initial_disparity",
        "initial_position",
        "initial_value",
        "expected",
    ],
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
            1,
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
            1,
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 1.5), np.float32(2.5), np.float32(1.5), np.float32(6.64453125)),
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
            1,
            0.25,
            (2.5, 1.5),
            (2.5, 1.5),
            7.638916,
            (refinement.dichotomy.Point(2.75, 1.25), np.float32(2.75), np.float32(1.25), np.float32(15.09161376953125)),
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
            1,
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
            1,
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
            1,
            0.5,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 2.5), np.float32(2.5), np.float32(2.5), np.float32(6.64453125)),
            id="NaN outside of kernel has no effect",
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
            2,
            0.25,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 1.5), np.float32(2.25), np.float32(1.75), np.float32(6.64453125)),
            id="Bottom left is best and subpix=2",
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
            2,
            0.125,
            (2, 2),
            (2, 2),
            1.0,
            (
                refinement.dichotomy.Point(2.25, 2.25),
                np.float32(2.125),
                np.float32(2.125),
                np.float32(1.77862548828125),
            ),
            id="Bottom right is best and subpix=2",
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
            4,
            0.125,
            (2, 2),
            (2, 2),
            1.0,
            (refinement.dichotomy.Point(2.5, 1.5), np.float32(2.125), np.float32(1.875), np.float32(6.64453125)),
            id="Bottom left is best and subpix=4",
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
            4,
            0.0625,
            (2, 2),
            (2, 2),
            1.0,
            (
                refinement.dichotomy.Point(2.25, 2.25),
                np.float32(2.0625),
                np.float32(2.0625),
                np.float32(1.77862548828125),
            ),
            id="Bottom right is best and subpix=4",
        ),
    ],
)
def test_search_new_best_point(
    make_cost_surface, filter_dicho, precision, initial_disparity, initial_position, initial_value, expected
):
    """Test we get new coordinates as expected."""

    cost_selection_method = np.nanargmax

    result = refinement.dichotomy.search_new_best_point(
        make_cost_surface,
        precision,
        initial_disparity,
        initial_position,
        initial_value,
        filter_dicho,
        cost_selection_method,
    )

    assert result == expected


@pytest.mark.parametrize(
    "dichotomy_instance_name",
    ["dichotomy_python_instance", "dichotomy_cpp_instance"],
)
class TestExtremaOnEdges:
    """
    Test that points for which best cost value is on the edge of disparity range
    are not processed by dichotomy loop.
    """

    @pytest.fixture()
    def left_img_non_uniform_grid(self, left_img):
        """
        Creates a left image dataset with non uniform disparity grids
        """

        # We set the minimum rows disparity at 4 for the point [0,1]
        left_img["row_disparity"][0, 0, 1] = 4
        # We set the maximum columns disparity at 0 for the point [1,0]
        left_img["col_disparity"][1, 1, 0] = 0

        return left_img

    @pytest.fixture()
    def cost_volumes(self, zeros_cost_volumes, min_disparity_row, max_disparity_col):
        """Build cost volumes."""
        # use indexes for row and col to be independent of coordinates which depend on ROI themselves,
        # but use coordinates for disp_row and disp_col

        # For point [0,2], the best cost value is set for minimal row disparity
        # corresponding cost surface is:

        #    [ 0.,  0.,  0.,  0.,  0.,  0.]
        #    [ 0.,  0.,  0.,  0.,  0.,  0.]
        #    [ 10., 8.,  0.,  0.,  0.,  0.]
        #    [ 8.,  9.,  0.,  0.,  0.,  0.]
        #    [ 0.,  0.,  0.,  0.,  0.,  0.]
        #    [ 0.,  0.,  0.,  0.,  0.,  0.]

        zeros_cost_volumes["cost_volumes"].isel(row=0, col=2).loc[
            {"disp_col": [0, 1], "disp_row": min_disparity_row + 1}
        ] = [8, 9]
        zeros_cost_volumes["cost_volumes"].isel(row=0, col=2).loc[
            {"disp_col": [0, 1], "disp_row": min_disparity_row}
        ] = [10, 8]

        # For point [0,1], the best cost value is set for row disparity greater than the minimal one
        # corresponding cost surface is:

        #   [ 0.,  0.,  0.,  0.,  0.,  0.]
        #   [ 0.,  0.,  0.,  0.,  0.,  0.]
        #   [ 0.,  8., 10.,  0.,  0.,  0.]
        #   [ 0.,  9.,  8.,  0.,  0.,  0.]
        #   [ 0.,  0.,  0.,  0.,  0.,  0.]
        #   [ 0.,  0.,  0.,  0.,  0.,  0.]

        zeros_cost_volumes["cost_volumes"].isel(row=0, col=1).loc[{"disp_col": [0, 1], "disp_row": 3}] = [8, 9]
        zeros_cost_volumes["cost_volumes"].isel(row=0, col=1).loc[{"disp_col": [0, 1], "disp_row": 4}] = [10, 8]

        # For point [0,0], the best cost value is set for maximal col disparity
        # corresponding cost surface is:

        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 4.9 , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 4.99, 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 5.  , 0.  , 0.  , 0.  ]

        zeros_cost_volumes["cost_volumes"].isel(row=0, col=0).loc[
            {"disp_col": [max_disparity_col - 2, max_disparity_col - 1, max_disparity_col], "disp_row": 4}
        ] = [
            4.9,
            4.99,
            5,
        ]

        # For point [1,0], the best cost value is set for col disparity lower than the maximal one
        # corresponding cost surface is:

        #    [0.  , 0.  , 4.9 , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 4.99, 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 5.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]

        zeros_cost_volumes["cost_volumes"].isel(row=1, col=0).loc[{"disp_col": [-2, -1, 0], "disp_row": 4}] = [
            4.9,
            4.99,
            5,
        ]

        return zeros_cost_volumes

    @pytest.fixture()
    def dataset_disp_maps(self, invalid_disparity, rows, cols, min_disparity_row, max_disparity_col, min_disparity_col):
        """Fake disparity maps containing extrema on edges of disparity range."""

        row = np.full((rows.size, cols.size), 4.0)
        row[:, 2] = min_disparity_row
        row[1, 1] = min_disparity_row

        # row map is equal to:
        # [4., 4., 2.]
        # [4., 2., 2.]

        col = np.full((rows.size, cols.size), 0.0)
        col[0, 0] = max_disparity_col
        col[1, -2:] = min_disparity_col

        # col map is equal to:
        # [3.,  0.,   0.]
        # [0., -2.,  -2.]

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

    def test_uniform_disparity_grid(
        self, request, cost_volumes, dataset_disp_maps, left_img, dichotomy_instance_name, mocker: MockerFixture
    ):
        """
        Test that points for which best cost value is on the edge of disparity range
        are not processed by dichotomy loop using uniform disparity grids
        """

        copy_disp_map = copy.deepcopy(dataset_disp_maps)

        dichotomy_instance = request.getfixturevalue(dichotomy_instance_name)
        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_img, img_right=mocker.ANY
        )

        # result_disp_row is equal to:
        # [4.   3.75 2.  ]
        # [4.   2.   2.  ]

        # result_disp_col is equal to:
        # [3.      0.25      0.  ]
        # [-0.25.  -2.      -2.  ]

        # Extrema on the edge of row disparity range for point [0,2] --> unchanged row map value after dichotomy loop
        assert result_disp_row[0, 2] == dataset_disp_maps["row_map"][0, 2]
        # Extrema not on the edge for point [0,1] --> changed row map value after dichotomy loop
        assert result_disp_row[0, 1] == dataset_disp_maps["row_map"][0, 1] - 0.25

        # Extrema on the edge of col disparity range for point [0,0] --> unchanged col map value after dichotomy loop
        assert result_disp_col[0, 0] == dataset_disp_maps["col_map"][0, 0]
        # Extrema not on the edge for point [1,0] --> changed col map value after dichotomy loop
        assert result_disp_col[1, 0] == dataset_disp_maps["col_map"][1, 0] - 0.25

    def test_non_uniform_disparity_grid(  # pylint: disable=too-many-arguments
        self,
        request,
        cost_volumes,
        dataset_disp_maps,
        left_img_non_uniform_grid,
        dichotomy_instance_name,
        max_disparity_row,
        min_disparity_col,
        mocker: MockerFixture,
    ):
        """
        Test that points for which best cost value is on the edge of disparity range
        are not processed by dichotomy loop using non uniform disparity grids
        """

        copy_disp_map = copy.deepcopy(dataset_disp_maps)

        dichotomy_instance = request.getfixturevalue(dichotomy_instance_name)
        result_disp_col, result_disp_row, _ = dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_img_non_uniform_grid, img_right=mocker.ANY
        )

        # result_disp_row is equal to:
        # [4.   4.   2.  ]
        # [4.   2.   2.  ]

        # result_disp_col is equal to:
        # [3.    0.     0. ]
        # [0.   -2.    -2. ]

        # Extrema on the edge of row disparity range for point [0,2] --> unchanged row map value after dichotomy loop
        assert result_disp_row[0, 2] == dataset_disp_maps["row_map"][0, 2]
        # Extrema on the edge of row disparity range for point [0,1] --> unchanged row map value after dichotomy loop
        assert result_disp_row[0, 1] == dataset_disp_maps["row_map"][0, 1]

        # For point [0,1] row disparity range is not [min_disparity_row, max_disparity_row] but [4, max_disparity_row],
        # we check that resulting disparity row is in this range.
        assert result_disp_row[0, 1] in range(4, max_disparity_row + 1)

        # Extrema on the edge of col disparity range for point [0,0] --> unchanged col map value after dichotomy loop
        assert result_disp_col[0, 0] == dataset_disp_maps["col_map"][0, 0]
        # Extrema on the edge of col disparity range for point [1,0] --> unchanged col map value after dichotomy loop
        assert result_disp_col[1, 0] == dataset_disp_maps["col_map"][1, 0]

        # For point [1,0] col disparity range is not [min_disparity_col, max_disparity_col] but [min_disparity_col, 0],
        # we check that resulting disparity row is in this range.
        assert result_disp_col[1, 0] in range(min_disparity_col, 0 + 1)


class TestChangeDisparityToIndex:

    @pytest.mark.parametrize(
        ["map", "shift", "subpixel", "expected"],
        [
            pytest.param(
                xr.DataArray(
                    data=[[1, 2, 3], [2, 2, 3]],
                    dims=("row", "col"),
                    coords={"row": np.arange(2), "col": np.arange(3)},
                ),
                1,  # disparity range starts at 1
                1,
                np.array([[0, 1, 2], [1, 1, 2]]),
                id="positive disparity",
            ),
            pytest.param(
                xr.DataArray(
                    data=[[-1, -2, -3], [-2, -2, -3]],
                    dims=("row", "col"),
                    coords={"row": np.arange(2), "col": np.arange(3)},
                ),
                -4,  # disparity range starts at -4
                1,
                np.array([[3, 2, 1], [2, 2, 1]]),
                id="negative disparity",
            ),
            pytest.param(
                xr.DataArray(
                    data=[[-1, -2.5, -3], [-2, -2.5, -3]],
                    dims=("row", "col"),
                    coords={"row": np.arange(2), "col": np.arange(3)},
                ),
                -4,  # disparity range starts at -4
                2,
                np.array([[6, 3, 2], [4, 3, 2]]),
                id="negative disparity and subpixel=0.5",
            ),
            pytest.param(
                xr.DataArray(
                    data=[[-1, -2.5, -3], [-2, -2.5, -3]],
                    dims=("row", "col"),
                    coords={"row": np.arange(2), "col": np.arange(3)},
                ),
                -4,  # disparity range starts at -4
                4,
                np.array([[12.0, 6, 4.0], [8, 6, 4]]),
                id="negative disparity and subpixel=0.25",
            ),
        ],
    )
    def test_disparity_to_index(self, map, shift, subpixel, expected):
        """Test disparity_to_index method"""
        result = refinement.dichotomy_cpp.disparity_to_index(map, shift, subpixel)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ["map", "shift", "subpixel", "expected"],
        [
            pytest.param(
                np.array([[0, 1, 2], [1, 1, 2]]),
                1,  # disparity range starts at 1
                1,
                np.array([[1, 2, 3], [2, 2, 3]]),
                id="positive disparity",
            ),
            pytest.param(
                np.array([[3, 2, 1], [2, 2, 1]]),
                -4,  # disparity range starts at -4
                1,
                np.array([[-1, -2, -3], [-2, -2, -3]]),
                id="negative disparity",
            ),
            pytest.param(
                np.array([[3.0625, 2, 1.0625], [2, 2, 1]]),
                -4,  # disparity range starts at -4 and precision = 0.0625
                2,
                np.array([[-2.46875, -3.0, -3.46875], [-3.0, -3.0, -3.5]]),
                id="negative disparity and subpixel=0.5",
            ),
            pytest.param(
                np.array([[3.0625, 2, 1.0625], [2, 2, 1]]),
                -4,  # disparity range starts at -4 and precision = 0.0625
                4,
                np.array([[-3.234375, -3.5, -3.734375], [-3.5, -3.5, -3.75]]),
                id="negative disparity and subpixel=0.5",
            ),
        ],
    )
    def test_index_to_disparity(self, map, shift, subpixel, expected):
        """Test index_to_disparity_method"""
        result = refinement.dichotomy_cpp.index_to_disparity(map, shift, subpixel)
        np.testing.assert_array_equal(result, expected)
