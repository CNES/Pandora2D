#!/usr/bin/env python
#
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
Test Disparity class
"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import xarray as xr
from rasterio import Affine
import json_checker

from pandora.margins import Margins
from pandora2d import matching_cost, disparity


class TestCheckConf:
    """Test check conf."""

    def test_nominal_case(self):
        """Should not raise error."""
        disparity.Disparity({"disparity_method": "wta", "invalid_disparity": -9999})

    def test_disparity_method_is_mandatory(self):
        """Should raise an error."""
        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError):
            disparity.Disparity({"invalid_disparity": "5"})

    def test_fails_with_bad_disparity_method_value(self):
        """Should raise an error."""
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            disparity.Disparity({"disparity_method": "WTN"})

    def test_default_invalid_disparity(self):
        result = disparity.Disparity({"disparity_method": "wta"})
        assert result.cfg["invalid_disparity"] == -9999

    def test_nan_invalid_disparity(self):
        result = disparity.Disparity({"disparity_method": "wta", "invalid_disparity": "NaN"})
        assert np.isnan(result.cfg["invalid_disparity"])


def test_margins():
    """
    test margins of matching cost pipeline
    """
    _disparity = disparity.Disparity({"disparity_method": "wta", "invalid_disparity": -9999})

    assert _disparity.margins == Margins(0, 0, 0, 0)


@pytest.mark.parametrize(
    ["extrema_func", "expected_result"],
    [
        pytest.param(
            np.max,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                    [[2.0, 3.0], [3.0, 4.0], [4.0, np.nan]],
                    [[2.0, 2.0], [3.0, 3.0], [4.0, np.nan]],
                ]
            ),
            id="test for maximum",
        ),
        pytest.param(
            np.min,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, np.nan]],
                    [[0.0, 1.0], [0.0, 1.0], [0.0, np.nan]],
                ]
            ),
            id="test for minimum",
        ),
    ],
)
def test_extrema_split(left_stereo_object, right_stereo_object, extrema_func, expected_result):
    """
    Test the min_split function
    """

    left, right = left_stereo_object, right_stereo_object

    # create a cost_volume, with SAD measure, window_size 1, dispx_min 0, dispx_max 1, dispy_min -1, dispy_max 0
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "sad", "window_size": 1}}}
    matching_cost_test = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    grid_min_col = np.full((3, 3), 0)
    grid_max_col = np.full((3, 3), 1)
    grid_min_row = np.full((3, 3), -1)
    grid_max_row = np.full((3, 3), 0)
    matching_cost_test.allocate_cost_volume_pandora(
        img_left=left, img_right=right, grid_min_col=grid_min_col, grid_max_col=grid_max_col, cfg=cfg
    )
    cvs = matching_cost_test.compute_cost_volumes(left, right, grid_min_col, grid_max_col, grid_min_row, grid_max_row)

    disparity_test = disparity.Disparity({"disparity_method": "wta", "invalid_disparity": -9999})
    # searching along dispy axis
    cvs_max = disparity_test.extrema_split(cvs, 3, extrema_func)

    np.testing.assert_allclose(cvs_max[:, :, 0], expected_result[:, :, 0], atol=1e-06)
    np.testing.assert_allclose(cvs_max[:, :, 1], expected_result[:, :, 1], atol=1e-06)


@pytest.mark.parametrize(
    ["extrema_func", "arg_extrema_func", "expected_result"],
    [
        pytest.param(
            np.max,
            np.argmax,
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
            id="test for maximum",
        ),
        pytest.param(
            np.min,
            np.argmin,
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
            id="test for minimum",
        ),
    ],
)
def test_arg_split(stereo_object_with_args, extrema_func, arg_extrema_func, expected_result):
    """
    Test the argmin_split function
    """

    left_arg, right_arg = stereo_object_with_args

    # create a cost_volume, with SAD measure, window_size 3, dispx_min 0, dispx_max 1, dispy_min -1, dispy_max 0
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "sad", "window_size": 3}}}

    matching_cost_test = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    grid_min_col = np.full((3, 3), 0)
    grid_max_col = np.full((3, 3), 1)
    grid_min_row = np.full((3, 3), -1)
    grid_max_row = np.full((3, 3), 0)
    matching_cost_test.allocate_cost_volume_pandora(
        img_left=left_arg,
        img_right=right_arg,
        grid_min_col=grid_min_col,
        grid_max_col=grid_max_col,
        cfg=cfg,
    )
    cvs = matching_cost_test.compute_cost_volumes(
        left_arg, right_arg, grid_min_col, grid_max_col, grid_min_row, grid_max_row
    )

    disparity_test = disparity.Disparity({"disparity_method": "wta", "invalid_disparity": -9999})
    # searching along dispy axis
    cvs_max = disparity_test.extrema_split(cvs, 3, extrema_func)
    min_tensor = disparity_test.arg_split(cvs_max, 2, arg_extrema_func)

    np.testing.assert_allclose(min_tensor, expected_result, atol=1e-06)


@pytest.mark.parametrize(
    "margins",
    [
        None,
        Margins(0, 0, 0, 0),
        Margins(0, 0, 1, 1),
        Margins(1, 1, 1, 1),
        pytest.param(
            Margins(3, 3, 3, 3),
        ),
        pytest.param(
            Margins(1, 2, 3, 4),
        ),
    ],
)
def test_compute_disparity_map_row(margins):
    """
    Test function for disparity computation
    """
    data = np.array(
        ([[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]]),
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    left = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "col_disparity_source": [-2, 2],
        "row_disparity_source": [-2, 2],
    }

    data = np.array(
        [[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]],
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    right = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    ground_truth_col = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    ground_truth_row = np.array([[2, 2, 2, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]])

    # create matching_cost object with measure = ssd, window_size = 1
    cfg_mc = {"pipeline": {"matching_cost": {"matching_cost_method": "ssd", "window_size": 1}}}

    matching_cost_matcher = matching_cost.MatchingCost(cfg_mc["pipeline"]["matching_cost"])
    # create disparity object with WTA method
    cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
    disparity_matcher = disparity.Disparity(cfg_disp)

    grid_min_col = np.full((3, 4), -2)
    grid_max_col = np.full((3, 4), 2)
    grid_min_row = np.full((3, 4), -2)
    grid_max_row = np.full((3, 4), 2)
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left,
        img_right=right,
        grid_min_col=grid_min_col,
        grid_max_col=grid_max_col,
        cfg=cfg_mc,
        margins=margins,
    )
    cvs = matching_cost_matcher.compute_cost_volumes(
        left, right, grid_min_col, grid_max_col, grid_min_row, grid_max_row, margins
    )

    delta_x, delta_y, _ = disparity_matcher.compute_disp_maps(cvs)

    np.testing.assert_array_equal(ground_truth_col, delta_x)
    np.testing.assert_array_equal(ground_truth_row, delta_y)


@pytest.mark.parametrize(
    "margins",
    [
        None,
        Margins(0, 0, 0, 0),
        Margins(1, 0, 1, 0),
        pytest.param(
            Margins(3, 3, 3, 3),
        ),
        pytest.param(
            Margins(1, 2, 3, 4),
        ),
    ],
)
def test_compute_disparity_map_col(margins):
    """
    Test function for disparity computation
    """
    data = np.array(
        ([[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]]),
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    left = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "col_disparity_source": [-3, 3],
        "row_disparity_source": [-3, 3],
    }

    data = np.array(
        [[8, 5, 6, 7], [4, 1, 2, 3], [12, 9, 10, 11]],
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    right = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    ground_truth_row = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    ground_truth_col = np.array([[1, 1, 1, -3], [1, 1, 1, -3], [1, 1, 1, -3]])

    # create matching_cost object with measure = ssd, window_size = 1
    cfg_mc = {"pipeline": {"matching_cost": {"matching_cost_method": "ssd", "window_size": 1}}}
    matching_cost_matcher = matching_cost.MatchingCost(cfg_mc["pipeline"]["matching_cost"])
    # create disparity object with WTA method
    cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
    disparity_matcher = disparity.Disparity(cfg_disp)

    grid_min_col = np.full((3, 4), -3)
    grid_max_col = np.full((3, 4), 3)
    grid_min_row = np.full((3, 4), -3)
    grid_max_row = np.full((3, 4), 3)
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left,
        img_right=right,
        grid_min_col=grid_min_col,
        grid_max_col=grid_max_col,
        cfg=cfg_mc,
        margins=margins,
    )
    cvs = matching_cost_matcher.compute_cost_volumes(
        left, right, grid_min_col, grid_max_col, grid_min_row, grid_max_row, margins
    )

    delta_x, delta_y, _ = disparity_matcher.compute_disp_maps(cvs)

    np.testing.assert_array_equal(ground_truth_col, delta_x)
    np.testing.assert_array_equal(ground_truth_row, delta_y)


@pytest.mark.parametrize(
    "margins",
    [
        None,
        Margins(0, 0, 0, 0),
        Margins(1, 0, 1, 0),
        pytest.param(
            Margins(3, 3, 3, 3),
        ),
        pytest.param(
            Margins(1, 2, 3, 4),
        ),
    ],
)
def test_compute_disparity_map_col_row(margins):
    """
    Test function for disparity computation
    """
    data = np.array(
        ([[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]]),
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    left = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "col_disparity_source": [-3, 3],
        "row_disparity_source": [-3, 3],
    }

    data = np.array(
        [[8, 5, 6, 7], [4, 1, 2, 3], [12, 9, 10, 11]],
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
    right = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    ground_truth_row = np.array([[2, 2, 2, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]])

    ground_truth_col = np.array([[1, 1, 1, -3], [1, 1, 1, -3], [1, 1, 1, -3]])

    # create matching_cost object with measure = ssd, window_size = 1
    cfg_mc = {"pipeline": {"matching_cost": {"matching_cost_method": "ssd", "window_size": 1}}}
    matching_cost_matcher = matching_cost.MatchingCost(cfg_mc["pipeline"]["matching_cost"])
    # create disparity object with WTA method
    cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
    disparity_matcher = disparity.Disparity(cfg_disp)

    grid_min_col = np.full((3, 4), -3)
    grid_max_col = np.full((3, 4), 3)
    grid_min_row = np.full((3, 4), -3)
    grid_max_row = np.full((3, 4), 3)
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left,
        img_right=right,
        grid_min_col=grid_min_col,
        grid_max_col=grid_max_col,
        cfg=cfg_mc,
        margins=margins,
    )
    cvs = matching_cost_matcher.compute_cost_volumes(
        left, right, grid_min_col, grid_max_col, grid_min_row, grid_max_row, margins
    )

    delta_x, delta_y, _ = disparity_matcher.compute_disp_maps(cvs)

    np.testing.assert_array_equal(ground_truth_col, delta_x)
    np.testing.assert_array_equal(ground_truth_row, delta_y)


def test_masked_nan():
    """
    Test the capacity of disparity_computation to find nans
    """
    cv = np.zeros((4, 5, 2, 2))
    # disp_x = -1, disp_y = -1
    cv[:, :, 0, 0] = np.array(
        [[np.nan, np.nan, np.nan, 6, 8], [np.nan, 0, 0, np.nan, 5], [1, 1, 1, 1, 1], [1, np.nan, 2, 3, np.nan]]
    )

    # disp_x = -1, disp_y = 0
    cv[:, :, 0, 1] = np.array(
        [[np.nan, np.nan, np.nan, 1, 2], [np.nan, 2, 2, 3, 6], [4, np.nan, 1, 1, 1], [6, 6, 6, 6, np.nan]]
    )

    # disp_x = 0, disp_y = 0
    cv[:, :, 1, 1] = np.array(
        [[np.nan, np.nan, np.nan, 0, 4], [np.nan, np.nan, 3, 3, 3], [2, np.nan, 4, 4, 5], [1, 2, 3, 4, np.nan]]
    )

    # disp_x = 0, disp_y = -1
    cv[:, :, 1, 0] = np.array(
        [[np.nan, np.nan, np.nan, 5, 60], [np.nan, 7, 8, 9, 10], [np.nan, np.nan, 6, 10, 11], [7, 8, 9, 10, np.nan]]
    )

    c_row = [0, 1, 2, 3]
    c_col = [0, 1, 2, 3, 4]

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(-1, 0 + 1)
    disparity_range_row = np.arange(-1, 0 + 1)

    cost_volumes_dataset = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
    )

    cost_volumes_dataset.attrs["type_measure"] = "max"
    cost_volumes_dataset.attrs["disparity_margins"] = None

    cfg_disp = {"disparity_method": "wta", "invalid_disparity": -99}
    disparity_matcher = disparity.Disparity(cfg_disp)

    ground_truth_col = np.array([[-99, -99, -99, -1, 0], [-99, 0, 0, 0, 0], [-1, -1, 0, 0, 0], [0, 0, 0, 0, -99]])

    ground_truth_row = np.array(
        [[-99, -99, -99, -1, -1], [-99, -1, -1, -1, -1], [0, -1, -1, -1, -1], [-1, -1, -1, -1, -99]]
    )

    delta_x, delta_y, _ = disparity_matcher.compute_disp_maps(cost_volumes_dataset)

    np.testing.assert_array_equal(ground_truth_col, delta_x)
    np.testing.assert_array_equal(ground_truth_row, delta_y)


@pytest.mark.parametrize(
    ["measure_type", "expected_score"],
    [
        pytest.param(np.min, np.array([[0.0, 5.0, 4.0], [3.0, 4.0, 5.0], [2.0, 1.0, 5.0]]), id="test for minimum"),
        pytest.param(
            np.max, np.array([[4.0, 88.0, 99.0], [66.0, 21.0, 52.0], [9.0, 8.0, 22.0]]), id="test for maximum"
        ),
    ],
)
def test_get_score(measure_type, expected_score):
    """
    test function for get_score
    """

    cost_volume = np.empty((3, 3, 3))

    cost_volume[:, :, 0] = np.array([[0, 5, 4], [3, 4, 8], [2, 1, 5]])
    cost_volume[:, :, 1] = np.array([[4, 88, 5], [3, 21, 5], [9, 8, 5]])
    cost_volume[:, :, 2] = np.array([[2, 7, 99], [66, 5, 52], [4, 5, 22]])

    cfg_disp = {"disparity_method": "wta", "invalid_disparity": -99}
    disparity_matcher = disparity.Disparity(cfg_disp)

    score = disparity_matcher.get_score(cost_volume, measure_type)

    assert np.array_equal(score, expected_score)
