#!/usr/bin/env python
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Test refinement step
"""

# pylint: disable=redefined-outer-name, protected-access
# mypy: disable-error-code=attr-defined


import numpy as np
import xarray as xr
import pytest

from json_checker.core.exceptions import DictCheckerError

from pandora.margins import Margins
from pandora2d import refinement, common


@pytest.fixture()
def cv_dataset():
    """
    Create dataset cost volumes
    """

    cv = np.zeros((3, 3, 5, 5))
    cv[:, :, 2, 2] = np.ones([3, 3])
    cv[:, :, 2, 3] = np.ones([3, 3])
    cv[:, :, 3, 2] = np.ones([3, 3])
    cv[:, :, 3, 3] = np.ones([3, 3])

    c_row = np.arange(cv.shape[0])
    c_col = np.arange(cv.shape[1])

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes_test = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
    )

    cost_volumes_test.attrs["measure"] = "zncc"
    cost_volumes_test.attrs["window_size"] = 1
    cost_volumes_test.attrs["type_measure"] = "max"

    return cost_volumes_test


@pytest.fixture()
def dataset_image():
    """
    Create an image dataset
    """
    data = np.arange(30).reshape((6, 5))

    img = xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    img.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "col_disparity_source": [-2, 2],
        "row_disparity_source": [-2, 2],
        "invalid_disparity": np.nan,
    }

    return img


def test_margins():
    """
    test margins of matching cost pipeline
    """
    _refinement = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]

    assert _refinement.margins == Margins(3, 3, 3, 3), "Not a cubic kernel Margins"


@pytest.mark.parametrize("refinement_method", ["interpolation", "optical_flow"])
def test_check_conf_passes(refinement_method):
    """
    Test the check_conf function
    """
    refinement.AbstractRefinement({"refinement_method": refinement_method})  # type: ignore[abstract]


@pytest.mark.parametrize(
    "refinement_config", [{"refinement_method": "wta"}, {"refinement_method": "optical_flow", "iterations": 0}]
)
def test_check_conf_fails(refinement_config):
    """
    Test the refinement check_conf with wrong configuration
    """

    with pytest.raises((KeyError, DictCheckerError)):
        refinement.AbstractRefinement(refinement_config)  # type: ignore[abstract]


def test_refinement_method_subpixel(cv_dataset):
    """
    test refinement_method with interpolation
    """

    cost_volumes_test = cv_dataset

    data = np.full((3, 3), 0.4833878)

    dataset_disp_map = common.dataset_disp_maps(data, data)

    test = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
    delta_x, delta_y = test.refinement_method(cost_volumes_test, dataset_disp_map, None, None)

    np.testing.assert_allclose(data, delta_y, rtol=1e-06)
    np.testing.assert_allclose(data, delta_x, rtol=1e-06)


def test_refinement_method_pixel(cv_dataset):
    """
    test refinement
    """

    cost_volumes_test = cv_dataset

    new_cv_datas = np.zeros((3, 3, 5, 5))
    new_cv_datas[:, :, 1, 3] = np.ones([3, 3])

    cost_volumes_test["cost_volumes"].data = new_cv_datas

    gt_delta_y = np.array(
        ([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
        dtype=np.float64,
    )

    gt_delta_x = np.array(
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        dtype=np.float64,
    )

    dataset_disp_map = common.dataset_disp_maps(gt_delta_y, gt_delta_x)

    test = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
    delta_x, delta_y = test.refinement_method(cost_volumes_test, dataset_disp_map, None, None)

    np.testing.assert_allclose(gt_delta_y, delta_y, rtol=1e-06)
    np.testing.assert_allclose(gt_delta_x, delta_x, rtol=1e-06)


def test_optical_flow_margins():
    """
    test get_margins of refinement pipeline
    """
    gt = (2, 2, 2, 2)  # with 5 has default window size
    _refinement = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]

    r_margins = _refinement.margins.astuple()

    assert r_margins == gt


def test_reshape_to_matching_cost_window_left(dataset_image):
    """
    Test reshape_to_matching_cost_window function for a left image
    """

    img = dataset_image

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]
    refinement_class._window_size = 3

    cv = np.zeros((6, 5, 5, 5))

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_col": disparity_range_col,
            "disp_row": disparity_range_row,
        },
    )

    # for left image
    reshaped_left = refinement_class.reshape_to_matching_cost_window(img, cost_volumes)

    # test four matching_cost
    idx_1_1 = [[0, 1, 2], [5, 6, 7], [10, 11, 12]]
    idx_2_2 = [[6, 7, 8], [11, 12, 13], [16, 17, 18]]
    idx_3_3 = [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
    idx_4_1 = [[15, 16, 17], [20, 21, 22], [25, 26, 27]]

    assert np.array_equal(reshaped_left[:, :, 0], idx_1_1)
    assert np.array_equal(reshaped_left[:, :, 4], idx_2_2)
    assert np.array_equal(reshaped_left[:, :, 8], idx_3_3)
    assert np.array_equal(reshaped_left[:, :, 9], idx_4_1)


def test_reshape_to_matching_cost_window_right(dataset_image):
    """
    Test reshape_to_matching_cost_window function for a right image
    """

    img = dataset_image

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]
    refinement_class._window_size = 3

    # Create disparity maps
    col_disp_map = [2, 0, 0, 0, 1, 0, 0, 0, 1, -2, 0, 0]
    row_disp_map = [2, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0]

    cv = np.zeros((6, 5, 5, 5))

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_col": disparity_range_col,
            "disp_row": disparity_range_row,
        },
    )

    # for right image
    reshaped_right = refinement_class.reshape_to_matching_cost_window(img, cost_volumes, row_disp_map, col_disp_map)

    # test four matching_cost
    idx_1_1 = [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
    idx_2_2 = [[2, 3, 4], [7, 8, 9], [12, 13, 14]]

    assert np.array_equal(reshaped_right[:, :, 0], idx_1_1)
    assert np.array_equal(reshaped_right[:, :, 4], idx_2_2)


def test_warped_image_without_step():
    """
    test warped image
    """

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]

    mc_1 = np.array(
        [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16], [18, 19, 20, 21, 22], [24, 25, 26, 27, 28]]
    )
    mc_2 = np.array(
        [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11], [13, 14, 15, 16, 17], [19, 20, 21, 22, 23], [25, 26, 27, 28, 29]]
    )

    reshaped_right = np.stack((mc_1, mc_2)).transpose((1, 2, 0))

    delta_row = -3 * np.ones(2)
    delta_col = -np.ones(2)

    test_img_shift = refinement_class.warped_img(reshaped_right, delta_row, delta_col, [0, 1])

    gt_mc_1 = np.array(
        [[19, 20, 21, 22, 22], [25, 26, 27, 28, 28], [25, 26, 27, 28, 28], [19, 20, 21, 22, 22], [13, 14, 15, 16, 16]]
    )

    gt_mc_2 = np.array(
        [[20, 21, 22, 23, 23], [26, 27, 28, 29, 29], [26, 27, 28, 29, 29], [20, 21, 22, 23, 23], [14, 15, 16, 17, 17]]
    )

    # check that the generated image is equal to ground truth
    assert np.array_equal(gt_mc_1, test_img_shift[:, :, 0])
    assert np.array_equal(gt_mc_2, test_img_shift[:, :, 1])
